def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs based on the fusion mode.
        
        Args:
            samples: Input samples containing visual features
            
        Returns:
            Tuple of (visual_outputs, visual_masks)
        """
        print("=" * 80)
        print("[prepare_visual_inputs] START")
        print("=" * 80)
        print(f"[prepare_visual_inputs] fusion_mode: {self.fusion_mode}")
        
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'
        
        print(f"[prepare_visual_inputs] spatial: {spatial}, spatiotemporal: {spatiotemporal}")
        print("-" * 80)

        # Process spatial features if needed
        if spatial:
            print("[prepare_visual_inputs] Processing SPATIAL features")
            print("-" * 80)
            print(f"[prepare_visual_inputs] samples['pixel_values'] length: {len(samples['pixel_values'])}")
            if len(samples['pixel_values']) > 0:
                print(f"[prepare_visual_inputs] samples['pixel_values'][0] shape: {samples['pixel_values'][0].shape}")
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            print(f"[prepare_visual_inputs] pixel_values shape: {pixel_values.shape}")
            spatial_outputs = self.spatio_proj(pixel_values)
            print(f"[prepare_visual_inputs] spatial_outputs shape: {spatial_outputs.shape}")
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
            print(f"[prepare_visual_inputs] spatial_mask shape: {spatial_mask.shape}")
            print("-" * 80)
        
        # Process spatiotemporal features if needed
        if spatiotemporal:
            print("[prepare_visual_inputs] Processing SPATIOTEMPORAL features")
            print("-" * 80)
            print(f"[prepare_visual_inputs] samples['glor_values'] length: {len(samples['glor_values'])}")
            if len(samples['glor_values']) > 0:
                print(f"[prepare_visual_inputs] samples['glor_values'][0] shape: {samples['glor_values'][0].shape}")
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            print(f"[prepare_visual_inputs] spatiotemporal_outputs (before proj) shape: {spatiotemporal_outputs.shape}")
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            print(f"[prepare_visual_inputs] spatiotemporal_outputs (after proj) shape: {spatiotemporal_outputs.shape}")
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
            print(f"[prepare_visual_inputs] spatiotemporal_mask shape: {spatiotemporal_mask.shape}")
            print("-" * 80)
        
        # Combine features for joint mode
        if self.fusion_mode == 'joint':
            print("[prepare_visual_inputs] JOINT MODE - Combining features")
            print("=" * 80)
            bs = spatial_outputs.shape[0]
            print(f"[prepare_visual_inputs] joint mode - batch size: {bs}")
            spatial_length = spatial_mask.sum(1)
            print(f"[prepare_visual_inputs] spatial_length shape: {spatial_length.shape}, values: {spatial_length}")
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            print(f"[prepare_visual_inputs] spatiotemporal_length shape: {spatiotemporal_length.shape}, values: {spatiotemporal_length}")
            new_length = spatial_length + spatiotemporal_length
            print(f"[prepare_visual_inputs] new_length shape: {new_length.shape}, values: {new_length}")
            print("-" * 80)
            
            # Concatenate spatial and spatiotemporal features for each sample
            print("[prepare_visual_inputs] Concatenating features per sample")
            print("-" * 80)
            joint_outputs = []
            for i in range(bs):
                valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                print(f"[prepare_visual_inputs] sample {i} - valid_spatial_output shape: {valid_spatial_output.shape}")
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                print(f"[prepare_visual_inputs] sample {i} - valid_spatiotemporal_output shape: {valid_spatiotemporal_output.shape}")
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                print(f"[prepare_visual_inputs] sample {i} - concat_sample shape: {concat_sample.shape}")
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            print(f"[prepare_visual_inputs] joint_outputs shape: {joint_outputs.shape}")
            print("-" * 80)
            
            # Apply temporal encoder
            print("[prepare_visual_inputs] Applying temporal encoder")
            print("-" * 80)
            joint_outputs_permuted = joint_outputs.permute(0,2,1)
            print(f"[prepare_visual_inputs] joint_outputs_permuted shape: {joint_outputs_permuted.shape}")
            new_length_tensor = torch.tensor(new_length.tolist(), device=self.device)
            print(f"[prepare_visual_inputs] new_length_tensor shape: {new_length_tensor.shape}, values: {new_length_tensor}")
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs_permuted, new_length_tensor
            )
            print(f"[prepare_visual_inputs] visual_conv_outputs keys: {visual_conv_outputs.keys()}")
            print(f"[prepare_visual_inputs] visual_conv_outputs['visual_feat'] shape: {visual_conv_outputs['visual_feat'].shape}")
            print(f"[prepare_visual_inputs] visual_conv_outputs['feat_len'] shape: {visual_conv_outputs['feat_len'].shape}, values: {visual_conv_outputs['feat_len']}")
            
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            print(f"[prepare_visual_inputs] visual_outputs shape: {visual_outputs.shape}")
            feat_len_list = visual_conv_outputs['feat_len'].to(torch.int).tolist()
            print(f"[prepare_visual_inputs] feat_len_list: {feat_len_list}")
            visual_masks = create_mask(
                seq_lengths=feat_len_list, 
                device=self.device
            )
            print(f"[prepare_visual_inputs] visual_masks shape: {visual_masks.shape}")
        else:
            # Use single feature type
            if spatial:
                print("=" * 80)
                print("[prepare_visual_inputs] SPATIAL MODE ONLY")
                print("=" * 80)
                spatial_outputs_permuted = spatial_outputs.permute(0,2,1)
                print(f"[prepare_visual_inputs] spatial_outputs_permuted shape: {spatial_outputs_permuted.shape}")
                num_frames_tensor = torch.tensor(samples['num_frames'], device=self.device)
                print(f"[prepare_visual_inputs] num_frames_tensor shape: {num_frames_tensor.shape}, values: {num_frames_tensor}")
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs_permuted, num_frames_tensor
                )
                print(f"[prepare_visual_inputs] spatial_conv_outputs keys: {spatial_conv_outputs.keys()}")
                print(f"[prepare_visual_inputs] spatial_conv_outputs['visual_feat'] shape: {spatial_conv_outputs['visual_feat'].shape}")
                print(f"[prepare_visual_inputs] spatial_conv_outputs['feat_len'] shape: {spatial_conv_outputs['feat_len'].shape}, values: {spatial_conv_outputs['feat_len']}")
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1,0,2)
                print(f"[prepare_visual_inputs] visual_outputs shape: {visual_outputs.shape}")
                feat_len_list = spatial_conv_outputs['feat_len'].to(torch.int).tolist()
                print(f"[prepare_visual_inputs] feat_len_list: {feat_len_list}")
                visual_masks = create_mask(
                    seq_lengths=feat_len_list, 
                    device=self.device
                )
                print(f"[prepare_visual_inputs] visual_masks shape: {visual_masks.shape}")
            elif spatiotemporal:
                print("=" * 80)
                print("[prepare_visual_inputs] SPATIOTEMPORAL MODE ONLY")
                print("=" * 80)
                visual_outputs = spatiotemporal_outputs
                print(f"[prepare_visual_inputs] visual_outputs shape: {visual_outputs.shape}")
                visual_masks = spatiotemporal_mask
                print(f"[prepare_visual_inputs] visual_masks shape: {visual_masks.shape}")
            else:
                raise NotImplementedError("Invalid fusion mode")
        
        print("=" * 80)
        print("[prepare_visual_inputs] FINAL RESULTS")
        print("=" * 80)
        print(f"[prepare_visual_inputs] visual_outputs shape: {visual_outputs.shape}, visual_masks shape: {visual_masks.shape}")
        print("=" * 80)
        print("[prepare_visual_inputs] END")
        print("=" * 80)
        return visual_outputs, visual_masks