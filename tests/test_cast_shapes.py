import torch

from seed_extension.models.cast.attention import CrossAttentionBlock, CrossAttentionDecoder, SeedSelfAttention
from seed_extension.models.cast.embedders import FourierEncode, HitEmbedder, SeedEmbedder


class TestFourierEncode:
    def test_output_shape(self):
        fe = FourierEncode(L=8, input_dim=3)
        x = torch.randn(10, 3)
        out = fe(x)
        assert out.shape == (10, 48)

    def test_normalization(self):
        fe = FourierEncode(L=4, input_dim=3)
        x = torch.randn(50, 3) * 100
        out = fe(x)
        assert torch.isfinite(out).all()
        assert out.min() >= -1.0 and out.max() <= 1.0


class TestHitEmbedder:
    def test_output_shape(self):
        emb = HitEmbedder(
            d_model=256,
            fourier_L=8,
            use_detector_features=True,
            use_cylindrical_pe=False,
        )
        x = torch.cat(
            [
                torch.randn(4, 100, 3),  # xyz
                torch.randint(0, 20, (4, 100, 1)),  # layer_id
                torch.randint(0, 14, (4, 100, 1)),  # volume_id
                torch.randint(0, 2, (4, 100, 1)),  # detector
            ],
            dim=-1,
        ).float()
        out = emb(x)
        assert out.shape == (4, 100, 256)

    def test_no_detector_features(self):
        emb = HitEmbedder(
            d_model=128,
            fourier_L=4,
            use_detector_features=False,
            use_cylindrical_pe=False,
        )
        x = torch.randn(2, 50, 3)
        out = emb(x)
        assert out.shape == (2, 50, 128)


class TestSeedEmbedder:
    def test_output_shape(self):
        emb = SeedEmbedder(d_model=256)
        x = torch.randn(4, 200, 9)
        out = emb(x)
        assert out.shape == (4, 200, 256)


class TestCrossAttentionBlock:
    def test_output_shape(self):
        block = CrossAttentionBlock(d_model=256, n_heads=8, ff_dim=1024, dropout=0.1)
        queries = torch.randn(4, 100, 256)
        kv = torch.randn(4, 1000, 256)
        out = block(queries, kv)
        assert out.shape == queries.shape

    def test_attention_scores_shape(self):
        block = CrossAttentionBlock(d_model=256, n_heads=8)
        queries = torch.randn(2, 50, 256)
        kv = torch.randn(2, 500, 256)
        out, attn = block(queries, kv, return_attention=True)
        assert attn.shape == (2, 50, 500)


class TestCrossAttentionDecoder:
    def test_output_shape(self):
        decoder = CrossAttentionDecoder(d_model=256, n_heads=8, ff_dim=1024, n_layers=3, dropout=0.1)
        queries = torch.randn(4, 100, 256)
        kv = torch.randn(4, 1000, 256)
        out = decoder(queries, kv)
        assert out.shape == queries.shape

    def test_return_attention(self):
        decoder = CrossAttentionDecoder(d_model=128, n_heads=4, ff_dim=512, n_layers=2, dropout=0.0)
        queries = torch.randn(2, 10, 128)
        kv = torch.randn(2, 100, 128)
        out, attn = decoder(queries, kv, return_attention=True)
        assert out.shape == queries.shape
        assert attn.shape == (2, 10, 100)


class TestSeedSelfAttention:
    def test_output_shape(self):
        sa = SeedSelfAttention(d_model=256, n_heads=8, dropout=0.1)
        x = torch.randn(4, 100, 256)
        out = sa(x)
        assert out.shape == x.shape
