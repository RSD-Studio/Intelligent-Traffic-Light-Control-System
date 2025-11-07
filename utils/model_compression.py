"""
Model Compression and Quantization Utilities
Reduces model size and speeds up inference
"""

import os
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class ModelCompressor:
    """Utilities for model compression and optimization"""

    @staticmethod
    def quantize_model(model, dtype=torch.qint8):
        """
        Quantize model to reduce size and speed up inference

        Args:
            model: PyTorch model to quantize
            dtype: Quantization dtype (torch.qint8 or torch.quint8)

        Returns:
            Quantized model
        """
        print("Quantizing model...")

        # Set model to eval mode
        model.eval()

        # Create a copy
        model_fp32 = deepcopy(model)

        # Fuse operations (Conv+BN+ReLU, etc.)
        # This is automatically handled for standard layers

        # Prepare for quantization
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model_fp32, inplace=True)

        # No calibration needed for dynamic quantization

        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_fp32, inplace=False)

        print("Quantization complete!")
        return quantized_model

    @staticmethod
    def dynamic_quantize(model):
        """
        Apply dynamic quantization (weights quantized, activations quantized at runtime)

        Args:
            model: PyTorch model

        Returns:
            Dynamically quantized model
        """
        print("Applying dynamic quantization...")

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8
        )

        print("Dynamic quantization complete!")
        return quantized_model

    @staticmethod
    def prune_model(model, amount=0.3):
        """
        Prune model weights to reduce size

        Args:
            model: PyTorch model
            amount: Fraction of weights to prune (0.0 to 1.0)

        Returns:
            Pruned model
        """
        print(f"Pruning model (amount: {amount})...")

        import torch.nn.utils.prune as prune

        # Prune all Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')  # Make pruning permanent

        print("Pruning complete!")
        return model

    @staticmethod
    def get_model_size(model):
        """
        Calculate model size in MB

        Args:
            model: PyTorch model

        Returns:
            Size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    @staticmethod
    def compare_models(original_model, compressed_model, sample_input):
        """
        Compare original and compressed models

        Args:
            original_model: Original PyTorch model
            compressed_model: Compressed model
            sample_input: Sample input tensor

        Returns:
            Dictionary with comparison metrics
        """
        print("\nComparing models...")

        # Get sizes
        orig_size = ModelCompressor.get_model_size(original_model)
        comp_size = ModelCompressor.get_model_size(compressed_model)

        # Get inference times
        import time

        original_model.eval()
        compressed_model.eval()

        with torch.no_grad():
            # Warm up
            for _ in range(10):
                _ = original_model(sample_input)
                _ = compressed_model(sample_input)

            # Time original
            start = time.time()
            for _ in range(100):
                orig_out = original_model(sample_input)
            orig_time = (time.time() - start) / 100

            # Time compressed
            start = time.time()
            for _ in range(100):
                comp_out = compressed_model(sample_input)
            comp_time = (time.time() - start) / 100

        # Calculate output difference
        output_diff = torch.abs(orig_out - comp_out).mean().item()

        results = {
            'original_size_mb': orig_size,
            'compressed_size_mb': comp_size,
            'size_reduction': (orig_size - comp_size) / orig_size * 100,
            'original_time_ms': orig_time * 1000,
            'compressed_time_ms': comp_time * 1000,
            'speedup': orig_time / comp_time,
            'output_difference': output_diff
        }

        print("\nComparison Results:")
        print(f"  Original Size: {orig_size:.2f} MB")
        print(f"  Compressed Size: {comp_size:.2f} MB")
        print(f"  Size Reduction: {results['size_reduction']:.1f}%")
        print(f"  Original Inference: {results['original_time_ms']:.3f} ms")
        print(f"  Compressed Inference: {results['compressed_time_ms']:.3f} ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Output Difference: {output_diff:.6f}")

        return results

    @staticmethod
    def save_compressed_model(model, filepath, compression_info=None):
        """
        Save compressed model with metadata

        Args:
            model: Compressed model
            filepath: Path to save
            compression_info: Dictionary with compression information
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'compression_info': compression_info or {}
        }

        torch.save(checkpoint, filepath)
        print(f"Compressed model saved to {filepath}")

    @staticmethod
    def knowledge_distillation(teacher_model, student_model, train_loader,
                               epochs=10, temperature=3.0, alpha=0.5,
                               device='cpu', learning_rate=0.001):
        """
        Train a smaller student model to mimic a larger teacher model

        Args:
            teacher_model: Large pre-trained model
            student_model: Smaller model to train
            train_loader: DataLoader with training data
            epochs: Number of training epochs
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss (1-alpha for student loss)
            device: Computing device
            learning_rate: Learning rate

        Returns:
            Trained student model
        """
        print("Starting knowledge distillation...")

        teacher_model.eval()
        student_model.train()

        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, (states, targets) in enumerate(train_loader):
                states = states.to(device)
                targets = targets.to(device)

                # Teacher predictions (soft targets)
                with torch.no_grad():
                    teacher_outputs = teacher_model(states)
                    soft_targets = nn.functional.softmax(teacher_outputs / temperature, dim=1)

                # Student predictions
                student_outputs = student_model(states)
                soft_predictions = nn.functional.log_softmax(student_outputs / temperature, dim=1)

                # Distillation loss
                distillation_loss = kl_loss(soft_predictions, soft_targets) * (temperature ** 2)

                # Student loss (on hard targets)
                student_loss = nn.functional.cross_entropy(student_outputs, targets)

                # Combined loss
                loss = alpha * distillation_loss + (1 - alpha) * student_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        print("Knowledge distillation complete!")
        return student_model


def compress_dqn_model(model_path, output_path, method='dynamic_quantization',
                       sample_state_size=15):
    """
    Compress a saved DQN model

    Args:
        model_path: Path to original model
        output_path: Path to save compressed model
        method: Compression method ('dynamic_quantization', 'pruning', or 'both')
        sample_state_size: Size of sample input state

    Returns:
        Compressed model and comparison results
    """
    # Load original model
    checkpoint = torch.load(model_path, map_location='cpu')

    # Reconstruct model (simplified - adjust based on actual model)
    from models.advanced_dqn import DuelingQNetwork

    config = checkpoint.get('config', {})
    state_size = config.get('state_size', sample_state_size)
    action_size = config.get('action_size', 2)
    hidden_size = config.get('hidden_size', 512)

    original_model = DuelingQNetwork(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        use_noisy=False,
        use_batch_norm=False
    )

    original_model.load_state_dict(checkpoint['q_network_state_dict'])
    original_model.eval()

    # Create sample input
    sample_input = torch.randn(1, state_size)

    # Apply compression
    compressor = ModelCompressor()

    if method == 'dynamic_quantization':
        compressed_model = compressor.dynamic_quantize(original_model)
    elif method == 'pruning':
        compressed_model = compressor.prune_model(deepcopy(original_model), amount=0.3)
    elif method == 'both':
        compressed_model = compressor.prune_model(deepcopy(original_model), amount=0.3)
        compressed_model = compressor.dynamic_quantize(compressed_model)
    else:
        raise ValueError(f"Unknown compression method: {method}")

    # Compare models
    comparison = compressor.compare_models(original_model, compressed_model, sample_input)

    # Save compressed model
    compressor.save_compressed_model(
        compressed_model,
        output_path,
        compression_info={
            'method': method,
            'comparison': comparison,
            'original_model_path': model_path
        }
    )

    return compressed_model, comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compress DQN Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to original model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save compressed model')
    parser.add_argument('--method', type=str, default='dynamic_quantization',
                       choices=['dynamic_quantization', 'pruning', 'both'],
                       help='Compression method')
    parser.add_argument('--state_size', type=int, default=15, help='State size')

    args = parser.parse_args()

    print("=" * 70)
    print("MODEL COMPRESSION")
    print("=" * 70)
    print(f"Original Model: {args.model_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Method: {args.method}")
    print("=" * 70)

    compressed_model, comparison = compress_dqn_model(
        args.model_path,
        args.output_path,
        method=args.method,
        sample_state_size=args.state_size
    )

    print("\n" + "=" * 70)
    print("COMPRESSION COMPLETE")
    print("=" * 70)
