# CNGE-v2 — Robust continuous geometric discovery

O fluxo recomendado agora usa **autoencoder diferencial + EKF**, **gauge escalar normalizado**, **dicionários funcionais + STLSQ** e **adaptadores LoRA com proteção de ativações ou roteamento por tarefa**. O pacote `cnge_v2/` contém todos os módulos da especificação v2, pipeline, checkpoints, demonstração e testes.

As fórmulas do pedido exigiram correções: `uᵀJu=0` não define uma norma, imagens de posição isoladas não identificam momento e subespaços finitos podem saturar. [Contratos, demonstrações e limites](docs/V2_VALIDATION.md) explicam exatamente o que o código garante e o que foi medido. O estado é pesquisa/desenvolvimento validado em CPU, sem alegar garantias universais de produção ou descoberta científica.

## Instalar

Python 3.11+ (validação local: Python 3.12). Use o checkout existente: cada tarefa cloud já é isolada, sem necessidade de worktree.

```bash
cd /workspace/CNGE
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip check
```

O PyTorch CPU está explícito para evitar baixar distribuições CUDA sem necessidade. Para GPU, instale o PyTorch adequado ao driver conforme a documentação oficial e configure `device="cuda"`. Nenhum serviço externo ou segredo é necessário para o v2.

## Executar e validar

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m cnge_v2.demo --output artifacts/v2-demo.json
OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q
.venv/bin/ruff check cnge_engine cnge_v2 tests tools
.venv/bin/ruff format --check cnge_engine cnge_v2 tests tools
.venv/bin/python -m build
```

Após instalação, `cnge-v2-demo` é equivalente ao módulo. A demonstração calibra coordenadas em telemetria ruidosa, executa ingestão → EKF → atualização LoRA → mineração e, separadamente, destila Runge–Lenz em cinco termos. Não apresenta o alvo analítico como uma descoberta neural autônoma.

A suíte normal inclui testes do v1 e v2. Três testes históricos de 50.000 passos exigem `pytest --long -m long -q`; a execução anterior levou cerca de 14 minutos. O v2 compartilha o mesmo midpoint validado, sem fallback Euler/RK4. [Resultados atuais](docs/V2_RESULTS.md) distinguem verificações atuais e históricas.

## Usar o pipeline

```python
import asyncio
import torch
from cnge_v2 import V2EngineConfig
from cnge_v2.execution.streaming_pipeline import StreamingEpistemicPipeline

async def run(calibration_pairs, sensor):
    # calibration_pairs: [B, 2, obs_dim], mesmo sensor e passo temporal do stream.
    engine = StreamingEpistemicPipeline(V2EngineConfig(
        obs_dim=calibration_pairs.shape[-1], latent_spatial_dim=2,
        device="cpu", protection_mode="task",
    ))
    engine.calibrate(calibration_pairs, steps=100)
    async for observation in sensor:
        telemetry = await engine.ingest_sensor_tick(observation)
        print(telemetry)
    engine.save_checkpoint("artifacts/model.pt")
    restored = StreamingEpistemicPipeline.load_checkpoint("artifacts/model.pt", device="cpu")
    return restored
```

Calibre com pares de observações igualmente espaçadas, normalizadas e representativas de várias trajetórias/energias. `Q_cov` e `R_cov` precisam refletir as unidades e o ruído do sensor. O encoder não é adaptado silenciosamente depois da calibração: uma mudança de carta invalidaria as garantias dos snapshots anteriores.

Para imagens de posição ou sensores sem estado observável em um frame, forme históricos com `ObservationHistory(frame_dim=128, length=4)` e configure `obs_dim=512`; `update(frame)` retorna `None` até preencher a janela, depois retorna o vetor concatenado. Prepare os pares de calibração com as mesmas janelas. Não se supõe que qualquer histórico arbitrário garanta observabilidade.

`LiveOrchestrator(engine, capacity=64, callback=...)` coordena um produtor e um worker com fila limitada, backpressure, cancelamento estruturado e propagação de erros. Aguarde `orchestrator.run(async_source)`. Não compartilhe o modelo com atualizações concorrentes fora desse worker.

### Aprendizado contínuo

- `protection_mode="activation"` é o padrão: protege a união dos subespaços capturados. Uma camada saturada gera `ActivationCapacityError`; expansão de características ou outra política exige uma decisão explícita.
- `protection_mode="task"` preserva snapshots nomeados e permite novos adaptadores completos. Após treinar um regime, use `engine.consolidate_task("regime_1")`; para inferência antiga, use `with engine.adapters.task("regime_1"):` ao avaliar energia/campo. Essa política requer contexto de tarefa e não promete proteção incondicional do campo ativo.
- `trigger_symbolic_distillation()` só consolida quando os gates de variância, comutador, R² separado e residual da fórmula passam. Retorna `None` quando não há evidência. Um sistema genérico com um grau de liberdade pode não ter outro invariante independente de H.

Os checkpoints preservam modelos, adaptadores e estados de otimização; filtros, filas e conexões reiniciam. Não recalibre a carta após congelar tarefas. A restauração de pesos não inicia serviços.

### Síntese analítica

`FunctionalDictionary` suporta monômios de grau limitado, trigonometria, filtros dimensionais/de escala e orçamento máximo de termos. `include_kepler=True` adiciona `q_i/r`, indispensável para Runge–Lenz. `ConstrainedSTLSQ.fit` aceita `constraints=C` para `C ξ=0`; `fit_gradient` ajusta gradientes de funções escalares, sem criar um campo não integrável. A complexidade é polinomial no tamanho explícito da biblioteca, que deve permanecer limitado.

## Código-fonte e distribuição

[CNGE_SOURCE.txt](CNGE_SOURCE.txt) reúne todo o conteúdo textual versionado, com caminhos e SHA-256: ambos os pacotes, testes, frontend, scripts, configuração e documentação. É um arquivo de fontes concatenadas, não um executável Python isolado. Após editar fontes:

```bash
git add .
.venv/bin/python tools/export_source.py
git add CNGE_SOURCE.txt
```

O build produz wheel e distribuição de fontes em `dist/`; esta inclui também ferramentas, interface e o TXT. Caches, ambientes virtuais, artefatos e credenciais não são exportados.

## Compatibilidade v1

O código `cnge_engine/` permanece disponível para comparação e compatibilidade. Ele não é chamado pelo pipeline v2 para estimar momento, buscar ASTs, minerar com a perda antiga ou projetar gradientes em parâmetros. O v2 reutiliza apenas suas primitivas matemáticas e ativações já verificadas. A interface e integrações opcionais antigas estão documentadas em [LEGACY_V1.md](docs/LEGACY_V1.md); o comando `tools/live_demo.py` continua sendo a demonstração **v1**, não a do pipeline novo.
