import os
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

class LoggingCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__()

    def log_generated_text(self, save_dir, ids, vis_strings, gloss_strings, generated_strings, reference_strings, prefix=None, logger=None, step=0):
        save_dir = os.path.join(save_dir, "text")
        os.makedirs(save_dir, exist_ok=True)
        file_name = "outputs.txt"
        
        if prefix is not None:
            file_name = f"{prefix}-outputs.txt"
        
        text_table = "| ID | Reference | Generated |\n|---|---|---|\n"

        if gloss_strings:
            with open(os.path.join(save_dir, file_name), "w") as file:
                for id, vis, gls, gen, ref in zip(ids, vis_strings, gloss_strings, generated_strings, reference_strings):
                    file.write(f"ID: {id}\nVis Token: {vis}\nGloss: {gls}\nReference: {ref}\nGenerated: {gen}\n\n")
                    text_table += f"| {id} | {ref} | {gen} |\n"
        else:
            with open(os.path.join(save_dir, file_name), "w") as file:
                for id, vis, gen, ref in zip(ids, vis_strings, generated_strings, reference_strings):
                    file.write(f"ID: {id}\nVis Token: {vis}\nReference: {ref}\nGenerated: {gen}\n\n")
                    text_table += f"| {id} | {ref} | {gen} |\n"

        if logger and isinstance(logger, TensorBoardLogger):
            logger.experiment.add_text(f"{prefix}_samples", text_table, step)

    def on_test_end(self, trainer, pl_module):
        ids = getattr(pl_module, 'id_list', [])
        vis_strings = getattr(pl_module, 'vis_string_list', [])
        glosses = getattr(pl_module, 'gloss_list', [])
        generated = getattr(pl_module, 'generated_text_list', [])
        references = getattr(pl_module, 'reference_text_list', [])

        if generated:
            self.log_generated_text(
                pl_module.logger.save_dir, ids, vis_strings, glosses, generated, references,
                prefix="test",
                logger=pl_module.logger,
                step=pl_module.global_step
            )

class MetricsTableCallback(Callback):
    def _log_table(self, trainer, name, filter_keyword=None):
        metrics = trainer.callback_metrics
        if not metrics:
            return

        table = "| Metric | Value |\n|---|---|\n"
        found_metrics = False
        
        for key in sorted(metrics.keys()):
            if filter_keyword and filter_keyword not in key:
                continue
                
            val = metrics[key]
            if isinstance(val, torch.Tensor):
                val = val.item()
            
            table += f"| {key} | {val:.6f} |\n"
            found_metrics = True
            
        if found_metrics and isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_text(name, table, trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_table(trainer, "Metrics_Table/Training", filter_keyword="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_table(trainer, "Metrics_Table/Validation", filter_keyword="val")

class MetricsPlotCallback(Callback):
    """Records the monitor metric (and loss) for train & val every epoch,
    then saves a matplotlib figure to the log directory after training ends."""

    def __init__(self, logdir: str, monitor: str):
        super().__init__()
        self.logdir = logdir
        self.monitor = monitor          # e.g. "val/bleu4" or "val/contra_loss"
        # Derive a paired train key from the monitor key, e.g. "train/bleu4"
        self.history: dict = defaultdict(list)  # key -> list of (epoch, value)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _record(self, trainer, keys):
        """Pull *keys* from callback_metrics and append to history."""
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        for key in keys:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.history[key].append((epoch, val))

    def _all_logged_keys(self, trainer):
        """Return every metric key currently in callback_metrics."""
        return list(trainer.callback_metrics.keys())

    # ------------------------------------------------------------------
    # per-epoch hooks
    # ------------------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        keys = [k for k in self._all_logged_keys(trainer) if k.startswith("train/")]
        self._record(trainer, keys)

    def on_validation_epoch_end(self, trainer, pl_module):
        keys = [k for k in self._all_logged_keys(trainer) if k.startswith("val/")]
        # Always attempt to capture the monitor metric
        if self.monitor not in keys:
            keys.append(self.monitor)
        self._record(trainer, keys)

    # ------------------------------------------------------------------
    # plot on training end
    # ------------------------------------------------------------------
    def on_train_end(self, trainer, pl_module):
        if not self.history:
            return
        try:
            self._save_plot(trainer)
        except Exception as e:
            print(f"[MetricsPlotCallback] Could not save plot: {e}")

    def _save_plot(self, trainer):
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        os.makedirs(self.logdir, exist_ok=True)

        # ---- collect val keys that have a matching train counterpart ----
        val_keys   = sorted(k for k in self.history if k.startswith("val/"))
        train_keys = sorted(k for k in self.history if k.startswith("train/"))

        # Determine which val metrics to plot (monitor first, then the rest)
        def _sort_key(k):
            return (0 if k == self.monitor else 1, k)
        val_keys = sorted(val_keys, key=_sort_key)

        n_metrics = max(len(val_keys), 1)
        fig, axes = plt.subplots(
            n_metrics, 1,
            figsize=(9, 4 * n_metrics),
            squeeze=False
        )
        fig.suptitle("Training Metrics", fontsize=14, fontweight="bold", y=1.01)

        def _unzip(pairs):
            if not pairs:
                return [], []
            epochs, vals = zip(*pairs)
            return list(epochs), list(vals)

        for ax, val_key in zip(axes[:, 0], val_keys):
            metric_name = val_key.split("/", 1)[-1]   # e.g. "bleu4"
            train_key   = f"train/{metric_name}"

            v_epochs, v_vals = _unzip(self.history.get(val_key, []))
            t_epochs, t_vals = _unzip(self.history.get(train_key, []))

            if v_vals:
                ax.plot(v_epochs, v_vals, marker="o", label=f"val/{metric_name}",
                        linewidth=2, color="#2196F3")
            if t_vals:
                ax.plot(t_epochs, t_vals, marker="s", label=f"train/{metric_name}",
                        linewidth=2, linestyle="--", color="#FF5722")

            # Highlight monitor metric
            if val_key == self.monitor and v_vals:
                best_idx = (v_vals.index(max(v_vals))
                            if "bleu" in metric_name or "rouge" in metric_name
                            else v_vals.index(min(v_vals)))
                ax.axvline(x=v_epochs[best_idx], color="green", linestyle=":",
                           linewidth=1.5, label=f"best epoch ({v_epochs[best_idx]})")
                ax.scatter([v_epochs[best_idx]], [v_vals[best_idx]],
                           color="green", zorder=5, s=80)

            title = metric_name.upper()
            if val_key == self.monitor:
                title += "  ★ monitor"
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # If there are train-only metrics (e.g. loss with no val counterpart)
        orphan_train = [k for k in train_keys
                        if f"val/{k.split('/', 1)[-1]}" not in val_keys]
        if orphan_train:
            ax_extra = axes[-1, 0] if len(val_keys) > 0 else axes[0, 0]
            for tk in orphan_train:
                t_epochs, t_vals = _unzip(self.history.get(tk, []))
                if t_vals:
                    ax_extra.plot(t_epochs, t_vals, marker="s",
                                  label=tk, linewidth=2, linestyle="--")
            ax_extra.legend(fontsize=9)
            ax_extra.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.logdir, "metrics_plot.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[MetricsPlotCallback] Saved metrics plot → {save_path}")

        # Also log to TensorBoard if available
        if isinstance(trainer.logger, TensorBoardLogger):
            try:
                import matplotlib.image as mpimg
                img = mpimg.imread(save_path)
                img_tensor = torch.tensor(img[:, :, :3]).permute(2, 0, 1).unsqueeze(0) / 255.0
                trainer.logger.experiment.add_images(
                    "metrics_plot", img_tensor, global_step=trainer.current_epoch
                )
            except Exception:
                pass


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            if pl_module.global_step != 0:
                print("[INFO] Summoning checkpoint.")
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))