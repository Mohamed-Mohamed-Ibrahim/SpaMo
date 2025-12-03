import os
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
    def _log_table(self, trainer, name):
        metrics = trainer.callback_metrics
        if not metrics:
            return

        table = "| Metric | Value |\n|---|---|\n"
        
        for key in sorted(metrics.keys()):
            val = metrics[key]
            if isinstance(val, torch.Tensor):
                val = val.item()
            table += f"| {key} | {val:.5f} |\n"
            
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_text(name, table, trainer.current_epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_table(trainer, "Metrics/Validation_Epoch")
        
    def on_test_end(self, trainer, pl_module):
        self._log_table(trainer, "Metrics/Test_Results")

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