            bs = spatial_outputs.shape[0]

            # --- DEBUG: Global Config & Inputs ---
            print(f"\n{'='*20} DEBUG START {'='*20}")
            print(f"[DEBUG] Batch Size: {bs}")
            print(f"[DEBUG] Device: {self.device}")
            print(f"[DEBUG] self.min_temporal_length: {self.min_temporal_length}")
            print(f"[DEBUG] self.inter_hidden: {self.inter_hidden}")
            print(f"[DEBUG] Input spatial_outputs: {spatial_outputs.shape} | dtype: {spatial_outputs.dtype}")
            print(f"[DEBUG] Input spatiotemporal_outputs: {spatiotemporal_outputs.shape} | dtype: {spatiotemporal_outputs.dtype}")
            # -------------------------------------

            spatial_length = spatial_mask.sum(1).cpu().tolist()
            spatiotemporal_length = spatiotemporal_mask.sum(1).cpu().tolist()

            # --- DEBUG: Raw Lengths ---
            print(f"[DEBUG] Raw spatial_lengths (first 5): {spatial_length[:5]}")
            print(f"[DEBUG] Raw spatiotemporal_lengths (first 5): {spatiotemporal_length[:5]}")
            # --------------------------
            
            aligned_outputs = []
            aligned_lengths = []
            
            for i in range(bs):
                s_len = int(spatial_length[i])
                st_len = int(spatiotemporal_length[i])
                
                # --- DEBUG: Inside Loop (Sample 0) ---
                if i == 0:
                    print(f"\n--- Processing Sample {i} ---")
                    print(f"[DEBUG] Initial s_len: {s_len}, st_len: {st_len}")
                # -------------------------------------

                # Check if either sequence is empty
                if s_len == 0 or st_len == 0:
                    if s_len > 0:
                        if i == 0: print("[DEBUG] Logic: Only Spatial valid. Fallback triggered.")
                        s_feat = spatial_outputs[i, :s_len, :]
                        st_feat = torch.zeros_like(s_feat)
                        aligned_len = s_len
                    elif st_len > 0:
                        if i == 0: print("[DEBUG] Logic: Only Spatiotemporal valid. Fallback triggered.")
                        st_feat = spatiotemporal_outputs[i, :st_len, :]
                        s_feat = torch.zeros_like(st_feat)
                        aligned_len = st_len
                    else:
                        if i == 0: print("[DEBUG] Logic: Both empty. Dummy creation triggered.")
                        aligned_len = 1
                        dummy_feat = torch.zeros(1, self.inter_hidden, device=self.device, dtype=spatial_outputs.dtype)
                        s_feat = dummy_feat
                        st_feat = dummy_feat
                else:
                    if i == 0: print("[DEBUG] Logic: Both valid. Aligning to min length.")
                    aligned_len = min(s_len, st_len)
                    s_feat = spatial_outputs[i, :aligned_len, :] 
                    st_feat = spatiotemporal_outputs[i, :aligned_len, :] 

                # --- DEBUG: Pre-Padding Shapes (Sample 0) ---
                if i == 0:
                    print(f"[DEBUG] After Slicing -> s_feat: {s_feat.shape}, st_feat: {st_feat.shape}, aligned_len: {aligned_len}")
                # --------------------------------------------
                
                # Ensure minimum length required for temporal encoder
                if aligned_len < self.min_temporal_length:
                    if i == 0: print(f"[DEBUG] Logic: aligned_len ({aligned_len}) < min_temporal ({self.min_temporal_length}). Padding required.")
                    
                    if aligned_len == 0:
                        dummy_feat = torch.zeros(self.min_temporal_length, self.inter_hidden, 
                                                device=self.device, dtype=spatial_outputs.dtype)
                        s_feat = dummy_feat
                        st_feat = dummy_feat
                        aligned_len = self.min_temporal_length
                    else:
                        n_repeats = self.min_temporal_length - aligned_len
                        if i == 0: print(f"[DEBUG] Repeating last frame {n_repeats} times.")
                        
                        last_frame_s = s_feat[-1:, :].repeat(n_repeats, 1)
                        last_frame_st = st_feat[-1:, :].repeat(n_repeats, 1)
                        s_feat = torch.cat([s_feat, last_frame_s], dim=0)
                        st_feat = torch.cat([st_feat, last_frame_st], dim=0)
                        aligned_len = self.min_temporal_length
                
                aligned_lengths.append(aligned_len)
                
                # --- DEBUG: Pre-Fusion Shapes (Sample 0) ---
                if i == 0:
                    print(f"[DEBUG] Ready for Fusion -> s_feat: {s_feat.shape}, st_feat: {st_feat.shape}")
                # -------------------------------------------

                # Apply adaptive fusion
                fused_feat = self.adaptive_fusion(
                    s_feat.unsqueeze(0), 
                    st_feat.unsqueeze(0) 
                )
                
                # --- DEBUG: Post-Fusion (Sample 0) ---
                if i == 0:
                    print(f"[DEBUG] fused_feat output: {fused_feat.shape}")
                # -------------------------------------

                aligned_outputs.append(fused_feat.squeeze(0)) 
            
            # Pad to same length for batch processing
            fused_outputs = pad_sequence(aligned_outputs, batch_first=True)
            
            valid_lengths = [max(length, self.min_temporal_length) for length in aligned_lengths]

            # --- DEBUG: Batch Aggregation ---
            print(f"\n--- Batch Aggregation ---")
            print(f"[DEBUG] aligned_lengths list (first 5): {aligned_lengths[:5]}")
            print(f"[DEBUG] valid_lengths list (passed to encoder, first 5): {valid_lengths[:5]}")
            print(f"[DEBUG] fused_outputs (padded batch) shape: {fused_outputs.shape}")
            # -------------------------------

            # Apply temporal encoder
            visual_conv_outputs = self.temporal_encoder(
                fused_outputs.permute(0,2,1), torch.tensor(valid_lengths, device=self.device)
            )
            
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2) 
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),
                device=self.device
            ) 
            
            # --- DEBUG: Final Output ---
            print(f"\n--- Final Outputs ---")
            print(f"[DEBUG] Encoder output 'visual_feat': {visual_conv_outputs['visual_feat'].shape}")
            print(f"[DEBUG] Encoder output 'feat_len' (first 5): {visual_conv_outputs['feat_len'][:5].tolist()}")
            print(f"[DEBUG] Final visual_outputs: {visual_outputs.shape}")
            print(f"[DEBUG] Final visual_masks: {visual_masks.shape}")
            print(f"{'='*20} DEBUG END {'='*20}\n")
            # ---------------------------