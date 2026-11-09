# CNGE — Continuous Neuro-Geometric Epistemic Engine

Implementação de referência executável da arquitetura neurogeométrica do anexo, em Python/PyTorch. Inclui os 22 módulos especificados, testes, demonstração científica, runtime de aprendizado online, transporte WebSocket e uma interface de inspeção em canvas.

**Estado:** pesquisa/desenvolvimento. As metas de descoberta autônoma geral, retenção perfeita, prova física, desempenho GPU e operação de produção ainda não estão demonstradas. Consulte [limites e correspondência com o pedido](docs/VALIDATION.md).

## Instalação

Python 3.11+ (validado localmente com Python 3.12). CPU não requer GPU, câmera, serviços externos ou credenciais:

```bash
cd /workspace/CNGE
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip check
```

Em uma máquina CUDA, instale primeiro a distribuição de PyTorch adequada ao driver usando as instruções oficiais do PyTorch, e selecione `EngineConfig(device="cuda")`. Não presuma disponibilidade de GPU. Cada tarefa cloud já é isolada: use o checkout existente, sem criar worktrees.

## Executar e verificar

```bash
.venv/bin/cnge-demo --steps 1000 --output artifacts/demo.json
.venv/bin/pytest -q
.venv/bin/pytest --long -m long -q  # três testes de 50.000 passos, execução mais demorada
.venv/bin/ruff check cnge_engine tests tools
.venv/bin/ruff format --check cnge_engine tests tools
.venv/bin/python -m build
```

A demonstração integra o oscilador harmônico isotrópico 3D, verifica um invariante analítico conhecido, sintetiza `L_z = q_x p_y - q_y p_x`, verifica a expressão em pontos separados e a registra na gramática. O relatório distingue esse exercício de uma descoberta neural autônoma.

## Interface e aprendizado online

```bash
.venv/bin/python tools/live_demo.py
# Execução finita, útil para smoke test:
.venv/bin/python tools/live_demo.py --steps 100
```

O processo inicia HTTP na porta 8081 e WebSocket na 8080, ambos em loopback. Para acesso local, abra o arquivo servido pelo HTTP no navegador da mesma máquina. A interface mostra `(q₁,q₂,p₁)`, métricas, candidatos e permite perturbações e restrição rotacional xy. Essa é uma projeção visual, não uma transformação simplética em 3D. Os processos não sobrevivem a snapshots: reinicie o comando quando necessário. `Ctrl+C` fecha o runtime e os servidores. Não exponha esse servidor sem autenticação e uma configuração de implantação própria.

Para outros sensores, forneça um iterador assíncrono de coordenadas `[d]` ao método `start_epistemic_runtime(source)` e aguarde a tarefa retornada. `stop()` cancela e coleta tarefas. A fonte deve ter período uniforme `integrator_dt` e massa constante conhecida para que a aproximação `p = m dq/dt` seja apropriada. Uma mudança de coordenadas observada não necessariamente corresponde a um impulso físico.

```python
import asyncio
from cnge_engine import EngineConfig
from cnge_engine.streaming.asynchronous_orchestrator import EpistemicContinuousOrchestrator

async def observations():
    for q in ([1., 0.], [.99995, .01], [.9998, .02]):
        yield q

async def main():
    engine = EpistemicContinuousOrchestrator(EngineConfig(spatial_dim=2))
    try:
        await engine.start_epistemic_runtime(observations())
        print(engine.metrics)
    finally:
        await engine.stop()

asyncio.run(main())
```

O runtime realiza atualização Hamiltoniana por SGD com gradiente projetado, mineração Noether por Adam, replay periódico, verificação empírica e compilação condicional. A mineração e as operações tensoriais são serializadas numa tarefa cooperativa para evitar corridas nos parâmetros; não se afirma execução concorrente em streams CUDA. A busca simbólica pode bloquear brevemente o loop. Eventos externos têm fila limitada e quadros são descartados em favor do mais recente para clientes lentos.

### Integrações opcionais

- **Embeddings:** `pip install -e '.[semantic]'`. `FoundationEmbeddingBridge` carrega um modelo Sentence Transformers explicitamente; o download requer acesso ao Hugging Face e seus hosts de armazenamento. Não ocorre na importação do pacote. O adaptador foi testado com backend injetado; o modelo remoto não foi baixado. A projeção contrastiva precisa de pares científicos rotulados antes de uso semântico significativo.
- **Câmera:** `pip install -e '.[camera]'`. `ContinuousSensorStream.capture(coordinate_extractor)` requer um dispositivo e uma função que transforme frames em coordenadas físicas calibradas. O código não interpreta pixels como posições ou inventa calibração.
- **Semântica:** a tradução de restrições suporta explicitamente rotação no plano xy. Prompts arbitrários são rejeitados; a explicação apresenta a expressão e o estado empírico, sem inventar citações ou provas.

## Estrutura

`cnge_engine/` segue a organização do pedido: `config`, `math_core`, `dynamics`, `noether`, `crystallization`, `semantic_bridge`, `streaming` e `visualization`. Os testes ficam em `tests/`; o frontend em `web/`. O pedido original está preservado em [docs/SPECIFICATION.txt](docs/SPECIFICATION.txt).

## Código-fonte em um único TXT

[CNGE_SOURCE.txt](CNGE_SOURCE.txt) reúne todos os arquivos de texto versionados (código, testes, interface, configuração e documentação), com caminho e SHA-256 de cada arquivo. É um arquivo de fontes concatenadas, não um executável binário. Recrie-o após alterar os fontes:

```bash
git add .
.venv/bin/python tools/export_source.py
```

O exportador exclui a si próprio como **saída** (`CNGE_SOURCE.txt`), mas inclui seu script, e não inclui `.venv`, caches, artefatos ou credenciais. A distribuição Python compilada/empacotada fica em `dist/` após `python -m build`.

### Teste do navegador

Com o servidor de demonstração ativo, execute `node tools/browser_smoke.cjs` em um ambiente com Playwright e Chromium. Se necessário, instale o cliente com `npm install --prefix /tmp/cnge-browser playwright` e use `NODE_PATH=/tmp/cnge-browser/node_modules node tools/browser_smoke.cjs`. `CHROMIUM_PATH` seleciona o executável Chromium instalado. O teste verifica telemetria real, intervenções, erros JavaScript e layout móvel.
