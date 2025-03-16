import torch
import pytest
from interpretability.sparse_autoencoder import SparseAutoencoder

@pytest.fixture
def sae():
    """Create a basic sparse autoencoder for testing"""
    return SparseAutoencoder(
        input_dim=10,
        hidden_dim=20,
        l1_coefficient=1e-3,
        use_top_k=False
    )

def test_initialization():
    """Test that the autoencoder initializes correctly"""
    input_dim, hidden_dim = 10, 20
    sae = SparseAutoencoder(input_dim, hidden_dim)
    
    # Check dimensions
    assert sae.encoder.weight.shape == (hidden_dim, input_dim)
    assert sae.decoder.weight.shape == (input_dim, hidden_dim)
    
    # Check decoder weights are normalized
    decoder_norms = torch.norm(sae.decoder.weight, dim=0)
    assert torch.allclose(decoder_norms, torch.ones_like(decoder_norms))

def test_forward_pass(sae):
    """Test the forward pass dimensions and basic functionality"""
    batch_size = 32
    x = torch.randn(batch_size, sae.input_dim)
    
    reconstructed, sparse_code = sae(x)
    
    # Check output dimensions
    assert reconstructed.shape == x.shape
    assert sparse_code.shape == (batch_size, sae.hidden_dim)
    
    # Check ReLU activation worked (no negative values)
    assert (sparse_code >= 0).all()

def test_top_k_sparsity():
    """Test that top-k sparsity works correctly"""
    sae = SparseAutoencoder(
        input_dim=10,
        hidden_dim=20,
        use_top_k=True,
        top_k_percentage=0.1
    )
    
    x = torch.randn(32, 10)
    _, sparse_code = sae(x)
    
    # Check that only 10% of neurons are active per sample
    active_per_sample = (sparse_code > 1e-8).sum(dim=1)
    expected_active = int(0.1 * sae.hidden_dim)
    assert all(active == expected_active for active in active_per_sample)

def test_dead_neuron_detection(sae):
    """Test that track_neuron_activity and identify_dead_neurons work correctly"""
    dead_idx = 5
    batch_size = 8
    
    # Create 10 batches of sparse codes where one neuron never activates above threshold
    for _ in range(10):
        # Create fake sparse codes with all neurons active
        sparse_code = torch.rand(batch_size, sae.hidden_dim) * 0.1 + 0.1  # All values > 0.1
        
        # Ensure the neuron at dead_idx has activations below the 1e-8 threshold
        sparse_code[:, dead_idx] = 1e-10  # Below activation threshold
        
        # Call the actual function being tested
        sae.track_neuron_activity(sparse_code)
    
    # Now check if identify_dead_neurons correctly identifies the dead neuron
    dead_neurons = sae.identify_dead_neurons()
    assert dead_neurons[dead_idx] == True
    assert dead_neurons.sum() == 1  # Only one dead neuron

def test_unit_norm_constraint(sae):
    """Test that gradient projection maintains unit norm"""
    x = torch.randn(32, sae.input_dim)
    x.requires_grad = True
    
    # Check initial norms are 1.0
    initial_norms = torch.norm(sae.decoder.weight.data, dim=0)
    assert torch.allclose(initial_norms, torch.ones_like(initial_norms))
    
    # Forward and backward pass
    reconstructed, sparse_code = sae(x)
    loss = torch.nn.functional.mse_loss(reconstructed, x)
    loss.backward()
    
    # Apply projection
    sae.project_decoder_grad()
    
    # Check that gradient components parallel to dictionary vectors are zero
    with torch.no_grad():
        for j in range(sae.hidden_dim):
            w_j = sae.decoder.weight.data[:, j].view(-1, 1)
            dw_j = sae.decoder.weight.grad.data[:, j].view(-1, 1)
            # Dot product should be close to zero
            parallel_component = torch.matmul(w_j.T, dw_j)
            assert torch.abs(parallel_component).item() < 1e-6
    
    # Simulate an optimizer step
    with torch.no_grad():
        lr = 0.01  # Small learning rate for testing
        sae.decoder.weight.data -= lr * sae.decoder.weight.grad.data
    
    # Check that norms are still close to 1.0 after the update
    updated_norms = torch.norm(sae.decoder.weight.data, dim=0)
    epsilon = 1e-4  # Small tolerance
    assert torch.all((updated_norms >= 1.0 - epsilon) & (updated_norms <= 1.0 + epsilon)), \
        f"Norms after update: min={updated_norms.min().item()}, max={updated_norms.max().item()}"

def test_neuron_resampling(sae):
    """Test that dead neurons are correctly resampled"""
    # Create fake activity history
    dead_idx = 5
    for _ in range(10):
        activity = torch.ones(sae.hidden_dim, dtype=torch.bool)
        activity[dead_idx] = False
        sae.neuron_activity_history.append(activity)
    
    # Create optimizer and sample inputs
    optimizer = torch.optim.Adam(sae.parameters())
    inputs = torch.randn(100, sae.input_dim)
    
    # Resample dead neurons
    loss_val, n_resampled = sae.resample_dead_neurons(inputs, optimizer)
    
    assert n_resampled == 1  # Should resample the one dead neuron
    assert torch.allclose(torch.norm(sae.decoder.weight[:, dead_idx]), torch.tensor(1.0))  # Should maintain unit norm



