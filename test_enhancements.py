"""
Quick Test Script to Verify All Enhancements
Run this to ensure all implementations are working correctly
"""

import sys
import os
import torch
import numpy as np

print("=" * 70)
print("TESTING ENHANCED IMPLEMENTATIONS")
print("=" * 70)

# Test 1: Import Enhanced Modules
print("\n[Test 1] Importing enhanced modules...")
try:
    from models.advanced_dqn import (
        ImprovedDQNAgent,
        DuelingQNetwork,
        LSTMQNetwork,
        PrioritizedReplayBuffer,
        NoisyLinear
    )
    from env.enhanced_sumo_env import EnhancedSumoEnv
    from scripts.enhanced_train import CurriculumTrainer
    from utils.model_compression import ModelCompressor
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create Enhanced Agent
print("\n[Test 2] Creating ImprovedDQNAgent...")
try:
    agent = ImprovedDQNAgent(
        state_size=15,
        action_size=2,
        hidden_size=256,
        use_double_dqn=True,
        use_dueling=True,
        use_noisy=True,
        use_per=True,
        use_lstm=False
    )
    print(f"✓ Agent created successfully")
    print(f"  - Device: {agent.device}")
    print(f"  - Double DQN: {agent.use_double_dqn}")
    print(f"  - Dueling: {agent.use_dueling}")
    print(f"  - Noisy: {agent.use_noisy}")
    print(f"  - PER: {agent.use_per}")
except Exception as e:
    print(f"✗ Agent creation failed: {e}")
    sys.exit(1)

# Test 3: Test Dueling Network Forward Pass
print("\n[Test 3] Testing DuelingQNetwork forward pass...")
try:
    sample_state = torch.randn(4, 15)  # Batch of 4 states
    q_values = agent.q_network(sample_state)
    assert q_values.shape == (4, 2), "Output shape mismatch"
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {sample_state.shape}")
    print(f"  - Output shape: {q_values.shape}")
    print(f"  - Sample Q-values: {q_values[0].detach().cpu().numpy()}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: Test LSTM Network
print("\n[Test 4] Testing LSTMQNetwork...")
try:
    lstm_agent = ImprovedDQNAgent(
        state_size=15,
        action_size=2,
        hidden_size=128,
        use_lstm=True
    )
    sample_state = torch.randn(4, 10, 15)  # Batch of 4, sequence length 10
    q_values, hidden = lstm_agent.q_network(sample_state)
    assert q_values.shape == (4, 2), "LSTM output shape mismatch"
    print(f"✓ LSTM network successful")
    print(f"  - Input shape: {sample_state.shape}")
    print(f"  - Output shape: {q_values.shape}")
except Exception as e:
    print(f"✗ LSTM test failed: {e}")
    sys.exit(1)

# Test 5: Test Prioritized Replay Buffer
print("\n[Test 5] Testing PrioritizedReplayBuffer...")
try:
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)

    # Add experiences
    for i in range(100):
        state = np.random.randn(15)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(15)
        done = False
        buffer.push(state, action, reward, next_state, done)

    # Sample batch
    states, actions, rewards, next_states, dones, indices, weights = buffer.sample(32)

    assert len(states) == 32, "Batch size mismatch"
    assert len(weights) == 32, "Weights size mismatch"

    print(f"✓ PER buffer working correctly")
    print(f"  - Buffer size: {len(buffer)}")
    print(f"  - Sample batch size: {len(states)}")
    print(f"  - Sample weights range: [{weights.min():.3f}, {weights.max():.3f}]")
except Exception as e:
    print(f"✗ PER buffer test failed: {e}")
    sys.exit(1)

# Test 6: Test Noisy Linear Layer
print("\n[Test 6] Testing NoisyLinear layer...")
try:
    noisy_layer = NoisyLinear(15, 64)
    sample_input = torch.randn(8, 15)

    # Training mode (with noise)
    noisy_layer.train()
    output_train = noisy_layer(sample_input)

    # Eval mode (no noise)
    noisy_layer.eval()
    output_eval = noisy_layer(sample_input)

    assert output_train.shape == (8, 64), "Noisy output shape mismatch"

    print(f"✓ Noisy layer working correctly")
    print(f"  - Output shape: {output_train.shape}")
    print(f"  - Training output mean: {output_train.mean().item():.3f}")
    print(f"  - Eval output mean: {output_eval.mean().item():.3f}")
except Exception as e:
    print(f"✗ Noisy layer test failed: {e}")
    sys.exit(1)

# Test 7: Test Agent Training Step
print("\n[Test 7] Testing agent training step...")
try:
    agent = ImprovedDQNAgent(
        state_size=15,
        action_size=2,
        hidden_size=128,
        batch_size=32
    )

    # Fill buffer
    for i in range(100):
        state = np.random.randn(15)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(15)
        done = False
        agent.remember(state, action, reward, next_state, done)

    # Training step
    loss = agent.replay()

    assert loss > 0, "Loss should be positive"

    print(f"✓ Training step successful")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Buffer size: {len(agent.memory)}")
except Exception as e:
    print(f"✗ Training step failed: {e}")
    sys.exit(1)

# Test 8: Test Curriculum Trainer
print("\n[Test 8] Testing CurriculumTrainer...")
try:
    curriculum = CurriculumTrainer(base_arrival_rate=0.25)

    stages = []
    while True:
        stage = curriculum.get_current_stage()
        if stage is None:
            break
        stages.append(stage['name'])
        curriculum.advance_stage()

    assert len(stages) == 4, "Should have 4 curriculum stages"

    print(f"✓ Curriculum trainer working correctly")
    print(f"  - Number of stages: {len(stages)}")
    print(f"  - Stage names: {stages}")
except Exception as e:
    print(f"✗ Curriculum trainer test failed: {e}")
    sys.exit(1)

# Test 9: Test Model Compression Utilities
print("\n[Test 9] Testing ModelCompressor...")
try:
    compressor = ModelCompressor()

    # Create simple model
    simple_model = torch.nn.Sequential(
        torch.nn.Linear(15, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    )

    # Get model size
    orig_size = compressor.get_model_size(simple_model)

    # Apply dynamic quantization
    quantized_model = compressor.dynamic_quantize(simple_model)
    quant_size = compressor.get_model_size(quantized_model)

    print(f"✓ Model compression working correctly")
    print(f"  - Original size: {orig_size:.3f} MB")
    print(f"  - Quantized size: {quant_size:.3f} MB")
    print(f"  - Reduction: {(1 - quant_size/orig_size)*100:.1f}%")
except Exception as e:
    print(f"✗ Model compression test failed: {e}")
    sys.exit(1)

# Test 10: Test Agent Save/Load
print("\n[Test 10] Testing model save/load...")
try:
    import tempfile

    agent = ImprovedDQNAgent(state_size=15, action_size=2, hidden_size=128)

    # Save model
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name

    agent.save_model(tmp_path)

    # Load model
    agent2 = ImprovedDQNAgent(state_size=15, action_size=2, hidden_size=128)
    agent2.load_model(tmp_path)

    # Clean up
    os.remove(tmp_path)

    print(f"✓ Save/load successful")
except Exception as e:
    print(f"✗ Save/load failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nYour enhanced system is ready to use!")
print("\nQuick start:")
print("  python main_enhanced.py --mode=train --episodes=50")
print("\nFor full documentation:")
print("  See README_ENHANCED.md and ENHANCEMENTS.md")
print("=" * 70)
