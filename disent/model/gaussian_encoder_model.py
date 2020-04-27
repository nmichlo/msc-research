import torch
from torch import Tensor


# ========================================================================= #
# gaussian encoder model                                                    #
# ========================================================================= #
from disent.model.encoders_decoders import DecoderSimpleConv64, EncoderSimpleConv64


class GaussianEncoderModel(torch.nn.Module):

    def __init__(self, gaussian_encoder: torch.nn.Module, decoder: torch.nn.Module):
        assert hasattr(gaussian_encoder, 'z_dim'), 'encoder does not define z_dim'
        assert hasattr(decoder, 'z_dim'), 'decoder does not define z_dim'
        assert gaussian_encoder.z_dim == decoder.z_dim, 'z_dim mismatch'
        super().__init__()
        self.gaussian_encoder = gaussian_encoder
        self.decoder = decoder
        self.z_dim = self.gaussian_encoder.z_dim

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        """
        reconstruct the input:
        x -> encode  | z_mean, z_var
          -> z       | z ~ p(z|x)
          -> decode  |
          -> x_recon | no final activation
        """
        # encode
        z_mean, z_logvar = self.encode_gaussian(x)  # .view(-1, self.x_dim)
        z = self.sample_from_latent_distribution(z_mean, z_logvar)
        # decode
        x_recon = self.decode(z).view(x.size())
        return x_recon, z_mean, z_logvar, z

    def forward_deterministic(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        # encode
        z_mean, z_logvar = self.encode_gaussian(x)  # .view(-1, self.x_dim)
        z = z_mean
        # decode
        x_recon = self.decode(z).view(x.size())
        return x_recon, z_mean, z_logvar, z

    @staticmethod
    def sample_from_latent_distribution(z_mean: Tensor, z_logvar: Tensor) -> Tensor:
        """
        Randomly sample for z based on the parametrization of the gaussian normal with diagonal covariance.
        This is an implementation of the 'reparametrisation trick'.
        ie. z ~ p(z|x)
        Gaussian Encoder Model Distribution - pg. 25 in Variational Auto Encoders
        """
        std = torch.exp(0.5 * z_logvar)  # std == var^0.5 == e^(log(var^0.5)) == e^(0.5*log(var))
        eps = torch.randn_like(std)      # N(0, 1)
        return z_mean + (std * eps)      # mu + dot(std, eps)

    def encode_gaussian(self, x: Tensor) -> (Tensor, Tensor):
        """
        Compute the mean and logvar parametrisation for the gaussian
        normal distribution with diagonal covariance ie. the parametrisation for p(z|x).
        """
        z_mean, z_logvar = self.gaussian_encoder(x)
        return z_mean, z_logvar

    def encode_stochastic(self, x):
        z_mean, z_logvar = self.encode_gaussian(x)
        return self.sample_from_latent_distribution(z_mean, z_logvar)

    def encode_deterministic(self, x):
        z_mean, z_logvar = self.encode_gaussian(x)
        return z_mean

    def decode(self, z: Tensor) -> Tensor:
        """
        Compute the partial reconstruction of the input from a latent vector.

        The final activation should not be included. This will always be sigmoid
        and is computed as part of the loss to improve numerical stability.
        """
        x_recon = torch.sigmoid(self.decoder(z))
        return x_recon

    def random_decoded_samples(self, num_samples):
        """
        Return a number of randomly generated reconstructions sampled from a gaussian normal input.
        """
        z = torch.randn(num_samples, self.z_dim)
        if torch.cuda.is_available():  # TODO: this is not gauranteed
            z = z.cuda()
        samples = self.decode(z)
        return samples

    # def reconstruct(self, z: Tensor) -> Tensor:
    #     """
    #     Compute the full reconstruction of the input from a latent vector.
    #     Like decode but performs a final sigmoid activation.
    #     """
    #     return torch.sigmoid(self.decode(z))

# ========================================================================= #
# END                                                                       #
# ========================================================================= #

