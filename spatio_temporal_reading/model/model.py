import torch.nn as nn
from spatio_temporal_reading.model.model_components import TransformerBlock


class SimpleModel(nn.Module):
    def __init__(
            self,
            model_type,
            d_in,
            d_model,
            n_layers,
            n_admixture_components
    ):
        super().__init__()

        self.model_type = model_type
        self.d_in = d_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_admixture_components = n_admixture_components
        self.initialize_submodules()

    def initialize_submodules(self):
        if self.model_type == "saccade":
            self.initialize_submodules_saccade()
        else:
            self.initialize_submodules_duration()

    def initialize_submodules_saccade(self):
        # Input of dimension (n_batches, len_sequence, d_in)

        # First embedding
        self.input_proj = nn.Linear(in_features=self.d_in, out_features=self.d_model)

        # Input to the attention layers of dimension (n_baches, len_sequence, d_model)
        # Then attention (keep it one-headed)
        self.attention_layers = nn.ModuleList([
            TransformerBlock(self.d_model)
            for _ in range(self.n_layers)
        ])

        # Finally, obtain the parameters of the distribution
        # Output of the model of dimension (n_batches, len_sequence, d_model)
        self.get_weights = nn.Linear(in_features=self.d_model, out_features=self.n_admixture_components)
        self.get_positions = nn.Linear(in_features=self.d_model, out_features = self.n_admixture_components * 2)
        self.get_saccades = nn.Linear(in_features=self.d_model, out_features=self.n_admixture_components)

    
    def initialize_submodules_duration(self):
        raise NotImplementedError
        
    
    def forward_saccades(self, x):
        # Input of dimension (n_batches, len_sequence, d_in)
        embeddings = self.input_proj(x) # (n_batches, len_sequence, d_model)

        # Go through the attention layers
        hidden_states = embeddings
        for attention_layer in self.attention_layers:
            hidden_states = attention_layer(hidden_states) # (n_batches, len_sequence, d_model)

        weights_scores = self.get_weights(hidden_states) # (n_batches, len_sequence, n_admixture_components)
        weights = weights_scores.softmax(dim=-1)
        positions = self.get_positions(hidden_states) # (n_batches, len_sequence, n_admixture_components * 2)
        B, T, _ = positions.shape
        K = self.n_admixture_components
        positions = positions.view(B, T, K, 2)
        saccades = self.get_saccades(hidden_states) # (n_batches, len_sequence, n_admixture_components)

        return weights, positions, saccades
    
    
    def forward_durations(self, input):
        raise NotImplementedError
    
    def forward(self, x):
        if self.model_type == "saccade":
            return self.forward_saccades(x)
        else:
            return self.forward_durations(x)
    
