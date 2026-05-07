import torch
import torch.nn as nn

class ModalityAligner(nn.Module):
    def __init__(self, d_audio: int, d_text: int = 771, d_model: int = 256):
        """
        d_audio : Dimension of your raw audio features (e.g., from Mel Spectrogram)
        d_text  : Dimension of the lyrics embeddings (771 from DeBERTa + VAD)
        d_model : The shared latent space dimension (d)
        """
        super().__init__()
        
        # Project lyrics from 771 to the shared latent dimension (d_model)
        self.proj_lyrics = nn.Linear(d_text, d_model)
        
        # Project audio from its original dimension to the shared latent dimension (d_model)
        self.proj_audio = nn.Linear(d_audio, d_model)

    def forward(self, E_audio: torch.Tensor, E_lyrics: torch.Tensor):
        """
        E_audio  : Tensor of shape (Batch, T, d_audio)
        E_lyrics : Tensor of shape (Batch, T, 771)
        
        Returns:
        X_audio  : Tensor of shape (Batch, T, d_model)
        X_lyrics : Tensor of shape (Batch, T, d_model)
        """
        
        # Apply the linear projections
        # X_lyrics = Linear(E_lyrics)
        X_lyrics = self.proj_lyrics(E_lyrics) 
        
        # X_audio = Linear(E_audio)
        X_audio = self.proj_audio(E_audio)     
        
        return X_audio, X_lyrics

# --- Quick Test ---
if __name__ == "__main__":
    batch_size = 4
    seq_length_T = 120 # Example number of windows
    d_audio = 128      # Example Mel Spectrogram feature dimension
    d_model = 256      # The shared 'd' dimension
    
    # Mock inputs
    mock_audio = torch.randn(batch_size, seq_length_T, d_audio)
    mock_lyrics = torch.randn(batch_size, seq_length_T, 771)
    
    aligner = ModalityAligner(d_audio=d_audio, d_text=771, d_model=d_model)
    
    X_audio, X_lyrics = aligner(mock_audio, mock_lyrics)
    
    print(f"X_audio shape: {X_audio.shape}")   # Expected: (4, 120, 256)
    print(f"X_lyrics shape: {X_lyrics.shape}") # Expected: (4, 120, 256)