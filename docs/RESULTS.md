# Resultados locais

Validação em CPU, Python 3.12, PyTorch 2.15.0+cpu. As versões efetivamente usadas estão em `VALIDATED_VERSIONS.txt`. Não havia CUDA disponível.

- Suíte rápida: **36 testes aprovados**. Os três testes longos são ignorados explicitamente nessa invocação e executados em separado.
- Suíte longa: **3 testes aprovados**, 150.000 passos no total, em **842,43 segundos**. Yoshida em Hénon–Heiles; midpoint em Hénon–Heiles e pêndulo duplo. Todos mantiveram o erro do mapa tangente simplético acumulado abaixo de **10⁻⁶**. A execução longa iniciou antes da adição de três testes rápidos; os 36 testes rápidos finais foram validados separadamente. Total: **39 testes aprovados**, sem falhas.
- `ruff check` e `ruff format --check`: aprovados.
- Compilação de bytecode (`compileall`) e `pip check`: aprovados.
- Distribuições `sdist` e `wheel`: geradas com sucesso; smoke test de um wheel instalado fora do checkout aprovado.
- Demonstração do oscilador 3D, 1.000 passos: drift relativo máximo de Lz **6,439293542825906 × 10⁻¹⁵**, expressão `z₀z₄ − z₁z₃`, **R² = 1,0** em pontos separados do ajuste. O alvo é um invariante analítico conhecido, não uma descoberta neural livre.
- Chromium com Playwright: telemetria WebSocket, aplicação de restrição xy, perturbação via canvas, ausência de erros JavaScript e layout móvel de 390 px aprovados. O teste reutilizável está em `tools/browser_smoke.cjs`.
- Smoke test de servidor finito: 20 observações e fechamento normal dos servidores.
- Benchmark indicativo de 100 atualizações após aquecimento: **135,60 atualizações/s**, CPU com uma thread de PyTorch, sem canvas e sem compilação simbólica. O tempo inclui aprendizado Hamiltoniano, mineração e integração. Não é uma garantia de desempenho, benchmark GPU ou medição do loop com todas as integrações ativas.
- Modelo Hamiltoniano padrão d=2: 33.793 parâmetros. Modelo Noether padrão: 8.705 parâmetros.

As integrações com câmera real, modelo fundacional baixado e CUDA não foram executadas. A CI foi configurada, mas uma execução remota só pode ser confirmada depois que o provedor a executar.

Condições externas verificadas: acesso Git de leitura e escrita ao GitHub funcionou. A criação de PR foi tentada, mas o proxy respondeu `Forbidden` ao endpoint `https://api.github.com/graphql`; o domínio precisa ser liberado nas configurações de rede do ambiente antes de repetir `gh pr create`. O acesso ao arquivo de configuração de `BAAI/bge-m3` no Hugging Face também recebeu HTTP 403 do proxy, e não havia dispositivos `/dev/video*` disponíveis. Nenhuma dessas verificações estabelece validação das integrações opcionais.
