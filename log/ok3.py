 def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs based on the fusion mode.
        
        Args:
            samples: Input samples containing visual features
            
        Returns:
            Tuple of (visual_outputs, visual_masks)
        """
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint', 'adaptive']:
            spatial = spatiotemporal = pose = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'
            pose = self.fusion_mode == 'pose'

        # Process spatial features if needed
        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            print(f"[spatial] pixel_values shape: {pixel_values.shape}")
            spatial_outputs = self.spatio_proj(pixel_values)
            print(f"[spatial] spatial_outputs shape: {spatial_outputs.shape}")
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
            print(f"[spatial] spatial_mask shape: {spatial_mask.shape}")
        
        # Process spatiotemporal features if needed
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            print(f"[spatiotemporal] padded glor_values shape: {spatiotemporal_outputs.shape}")
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            print(f"[spatiotemporal] spatiotemporal_outputs shape: {spatiotemporal_outputs.shape}")
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
            print(f"[spatiotemporal] spatiotemporal_mask shape: {spatiotemporal_mask.shape}")
        
        # Process pose features if needed
        if pose:
            raw_pose_values = samples.get('pose_values', [])
            print(f"[pose] raw_pose_values length: {len(raw_pose_values)}")
            pose_values_local = [pv if pv.dim() == 2 else pv.view(pv.shape[0], -1) for pv in raw_pose_values]
            if len(pose_values_local) > 0:
                pose_padded = pad_sequence(pose_values_local, batch_first=True).to(self.device).float()
                print(f"[pose] pose_padded shape: {pose_padded.shape}")
                pose_lengths = [int(p.size(0)) for p in pose_values_local]
            else:
                B = len(samples['pixel_values'])
                pose_padded = torch.zeros((B, 1, self.pose_input_size), device=self.device, dtype=torch.float32)
                print(f"[pose] pose_padded (empty case) shape: {pose_padded.shape}")
                pose_lengths = [0] * B
            pose_outputs = self.pose_proj(pose_padded)
            print(f"[pose] pose_outputs shape: {pose_outputs.shape}")
            pose_mask = create_mask(seq_lengths=pose_lengths, device=self.device)
            print(f"[pose] pose_mask shape: {pose_mask.shape}")
        
        # Combine features for joint mode
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            pose_length = pose_mask.sum(1) if pose else torch.zeros_like(spatial_length)
            print(f"[joint] batch_size: {bs}, spatial_length shape: {spatial_length.shape}, spatiotemporal_length shape: {spatiotemporal_length.shape}, pose_length shape: {pose_length.shape}")
            new_length = spatial_length + spatiotemporal_length + pose_length
            print(f"[joint] new_length shape: {new_length.shape}, new_length: {new_length}")

            # Concatenate spatial, spatiotemporal and pose features for each sample
            joint_outputs = []
            for i in range(bs):
                parts = []
                if spatial:
                    parts.append(spatial_outputs[i, :spatial_length[i], :])
                if spatiotemporal:
                    parts.append(spatiotemporal_outputs[i, :spatiotemporal_length[i], :])
                if pose:
                    parts.append(pose_outputs[i, :pose_length[i], :])
                concat_sample = torch.cat(parts, dim=0)
                print(f"[joint] sample {i}: concat_sample shape: {concat_sample.shape}")
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            print(f"[joint] final joint_outputs shape: {joint_outputs.shape}")

            # Apply temporal encoder
            print(f"[joint] input to temporal_encoder - shape: {joint_outputs.permute(0,2,1).shape}, lengths: {new_length.tolist()}")
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
            )

            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            print(f"[joint] visual_outputs shape: {visual_outputs.shape}")
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), 
                device=self.device
            )
            print(f"[joint] visual_masks shape: {visual_masks.shape}")