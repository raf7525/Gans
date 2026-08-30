# Gans

Implementação de um GAN (Generative Adversarial Network) simples para apresentação de faculdade.

Duas redes neurais competem entre si:
- **Gerador (G)**: recebe ruído aleatório e tenta produzir dígitos falsos que pareçam reais.
- **Discriminador (D)**: recebe uma imagem (real ou falsa) e tenta dizer qual é qual.

Treinadas juntas, ao final o Gerador sozinho consegue criar dígitos novos plausíveis a partir de puro ruído.

## Uso

```bash
pip install -r requirements.txt
python3 gan_mnist.py --epochs 30
```

O MNIST é baixado automaticamente na primeira execução. Resultados salvos em `output/`:
- `samples_epoch_XXX.png` — grade de dígitos gerados a cada N épocas
- `loss_curve.png` — curva de loss do Gerador e do Discriminador
- `generator.pt` — pesos do gerador treinado

Parâmetros disponíveis: `--epochs`, `--batch-size`, `--lr`, `--sample-every`.
