# Escopo científico e validação

## Contratos implementados

| Área | Implementação | Verificação |
|---|---|---|
| Geometria canônica | J como buffer, campos Hamiltonianos, Poisson autodiff, teste de simpléticidade | Colchetes canônicos d=1,2,3, potencial central, Jacobianos não lineares |
| Integração | Midpoint implícito diferenciável com falha explícita de convergência; Yoshida separável | Reversibilidade, mapas tangentes cumulativos, erro de estrutura |
| Álgebra de Lie | JVPs, mínimos quadrados de constantes, derivada de Lie da métrica | Translação/dilatação e campo rotacional de Killing |
| Modelos H/L | MLPs suaves, resíduos dinâmicos, Hessiana Euler–Lagrange | Aprendizado com redução de loss, aceleração analítica e gradientes de parâmetros |
| Memória | Sensibilidades de invariantes e projeção sem matriz P×P | Resíduo nulo sem damping e redução com damping, inclusive vínculos dependentes |
| Noether | Objetivo com comutador, variância e ortogonalidade | Gradientes finitos e redução de objetivo; não se afirma convergência única |
| Conservação | Máximo drift relativo e limite Hoeffding separado para falhas IID | Rejeição de drift/NaN; demonstração conservativa |
| Topologia | Seção de Poincaré interpolada, dimensão de correlação e métrica pullback | Órbita circular e fluxo linear |
| Síntese | AST segura sem eval, busca limitada, ajuste quadrático, fronteira de complexidade/erro, registro dinâmico | Lz exato, R² em dados separados, aridade e registro |
| Semântica | Adaptador de modelo opcional, projeção InfoNCE, vocabulário explícito xy | Backend injetado, backward contrastivo, gerador canônico |
| Streaming | Buffer limitado, diferenças finitas/EMA, runtime cancelável, replay e intervenções | Aprendizado altera pesos, fila, cancelamento, filtro |
| Visualização | Slice 3D, órbitas de geradores, WebSocket e canvas | Geometria gerada, roundtrip real, inspeção automatizada do navegador |

## Correções e limites da especificação

1. Simpléticidade não significa conservação exata da energia nem estabilidade por tempo infinito. Há erro de discretização, arredondamento e convergência; o midpoint ajusta sua tolerância mínima à precisão do tensor. Os testes usam float64. Verificações em um estado inicial de baixa energia não provam todos os regimes caóticos.
2. O pêndulo duplo canônico não tem Hamiltoniano separável `T(p)+V(q)`. Ele é validado com midpoint; executar Yoshida separável nesse modelo seria matematicamente incorreto. O teste Yoshida de 50.000 passos usa Hénon–Heiles. Os testes mantêm o mapa tangente cumulativo, sem confundi-lo com Jacobianos isolados.
3. O objetivo Noether tem vários mínimos e degenerações. O teste de aprendizado exige redução mensurável, não finge que a otimização tem uma solução única Lz. O teste de cristalização recebe o Lz analítico como alvo. A descoberta neural irrestrita solicitada permanece uma meta de pesquisa não demonstrada.
4. A penalização de variância não garante não trivialidade nem independência funcional. Um invariante varia entre órbitas, mas é constante numa órbita individual. O runtime conserva um buffer de uma sequência e pode corretamente não produzir nenhuma descoberta. Pontos intercalados de uma mesma trajetória não são amostras IID nem comprovam generalização física. A saída é um candidato empírico.
5. A projeção em parâmetros é Euclidiana; não há forma simplética canônica dada sobre os pesos. Com damping positivo ela não é uma projeção ortogonal exata. Mesmo sem damping a restrição é apenas local e de primeira ordem; médias de sensibilidades podem cancelar e a base fica desatualizada. Não se comprovou retenção de Runge–Lenz após aprender um sistema quártico. Os testes não substituem esse experimento por uma alegação equivalente.
6. A busca simbólica é finita e pode não alcançar o R² solicitado; o chamador deve verificar fidelidade. A fronteira é calculada sobre candidatos visitados e usa contagem de nós como proxy de descrição, não um mínimo global de Kolmogorov. Ajustes polinomiais precisam de domínios suficientemente informativos. Operações fora do domínio real são descartadas sem mudar a fórmula.
7. Uma seção genérica de Poincaré não possui automaticamente coordenadas intrínsecas: por padrão retornam-se interseções em 2d; `chart_fn` fornece o mapa para 2d−1. A estimativa de dimensão é finita e depende das escalas e da amostragem; não demonstra existência de atratores dissipativos em sistemas Hamiltonianos.
8. Projeção contrastiva entre espaços de dimensões diferentes não é um difeomorfismo. Não se implementa um solver Gromov–Wasserstein nem recuperação automática de literatura. Embeddings sem treinamento pareado não identificam conceitos físicos por si sós.
9. Projeção para dimensão 3 é exclusivamente visual. Não existe forma simplética não degenerada em dimensão ímpar.
10. O filtro `p≈m dq/dt` pressupõe massa constante, coordenadas calibradas e intervalos uniformes. Não é um filtro canônico universal, especialmente no pêndulo duplo. A primeira amostra não entra no treinamento, pois não possui estimativa de velocidade.
11. Não há benchmark RTX, validação CUDA, captura de câmera real, modelo fundacional remoto, WebRTC, provas físicas, autenticação multiusuário ou SLA de produção nesta entrega. O canvas redesenha até a taxa do navegador e o servidor limita envio a 60 Hz; isso não comprova 60 FPS ou 120 atualizações/s. Os parâmetros e recursos dependem de dimensões/modelos escolhidos, não são tratados como orçamento garantido.
12. A configuração padrão serve a CPU. As redes especificadas somam mais parâmetros do que os números indicados no orçamento do anexo; o código prioriza as assinaturas e dimensões solicitadas, sem alegar que respeita o total estimado de 200 mil.

## Reprodução das evidências

Execute os comandos do README. A suíte normal marca explicitamente os três testes longos como ignorados se `--long` estiver ausente. `pytest --long -m long` executa 150.000 passos no total. A CI padrão executa a suíte rápida, lint, demonstração, empacotamento e comparação byte a byte do TXT; os testes longos podem ser executados separadamente.

A aplicação de referência e as integrações opcionais têm estados de validação diferentes. Nenhum teste com backend semântico injetado deve ser interpretado como teste de um modelo fundacional baixado.
