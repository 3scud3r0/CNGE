# Resultados locais

Validação em CPU, Python 3.12, PyTorch 2.15.0+cpu. As versões efetivamente usadas estão em `VALIDATED_VERSIONS.txt`. Não havia CUDA disponível.

- Suíte rápida: **36 testes aprovados**. Os três testes longos são ignorados explicitamente nessa invocação e executados em separado.
- `ruff check` e `ruff format --check`: aprovados.
- Compilação de bytecode (`compileall`) e `pip check`: aprovados.
- Distribuições `sdist` e `wheel`: geradas com sucesso; smoke test de um wheel instalado fora do checkout aprovado.
- Demonstração do oscilador 3D, 1.000 passos: drift relativo máximo de Lz **6,439293542825906 × 10⁻¹⁵**, expressão `z₀z₄ − z₁z₃`, **R² = 1,0** em pontos separados do ajuste. O alvo é um invariante analítico conhecido, não uma descoberta neural livre.
- Chromium com Playwright: telemetria WebSocket, aplicação de restrição xy, perturbação via canvas, ausência de erros JavaScript e layout móvel de 390 px aprovados. O teste reutilizável está em `tools/browser_smoke.cjs`.
- Smoke test de servidor finito: 20 observações e fechamento normal dos servidores.
- Benchmark indicativo de 100 atualizações após aquecimento: **135,60 atualizações/s**, CPU com uma thread de PyTorch, sem canvas e sem compilação simbólica. O tempo inclui aprendizado Hamiltoniano, mineração e integração. Não é uma garantia de desempenho, benchmark GPU ou medição do loop com todas as integrações ativas.
- Modelo Hamiltoniano padrão d=2: 33.793 parâmetros. Modelo Noether padrão: 8.705 parâmetros.

As integrações com câmera real, modelo fundacional baixado e CUDA não foram executadas. A CI foi configurada, mas uma execução remota só pode ser confirmada depois que o provedor a executar.
