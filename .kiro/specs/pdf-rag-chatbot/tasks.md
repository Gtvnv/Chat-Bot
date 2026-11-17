# Implementation Plan: PDF RAG Chatbot (Plano de Implementação)

## Overview

A implementação segue o design em Python 3.11+, como CLI interativa (`python -m pdf_rag_chatbot`), com testes em `pytest` + `hypothesis`. A ordem das tarefas vai do núcleo puro (erros, modelos, configuração, chunking) para os adaptadores (embeddings, índice, provedores de geração), depois para a orquestração (Ingestor, Recuperador, Chatbot), a CLI e, por fim, os artefatos de entrega do repositório (`inputs/`, README, licença). Cada etapa termina integrada à anterior, sem código órfão.

Todo o código, os comentários, os nomes de domínio e as mensagens ao usuário ficam em português, seguindo a nomenclatura do design (`Ingestor`, `Recuperador`, `Trecho`, `Citacao` e assim por diante).

## Tasks

- [ ] 1. Preparar a estrutura do projeto e a base de testes
  - [ ] 1.1 Criar a estrutura do repositório e os arquivos de configuração
    - Criar `src/pdf_rag_chatbot/` (com `__init__.py`, `__main__.py` mínimo), `tests/`, `inputs/` e `docs/img/`
    - Criar `pyproject.toml` com layout `src`, `pythonpath` do pytest e `testpaths = ["tests"]`
    - Criar `requirements.txt` (`pypdf>=4.0`, `numpy>=1.26`, `httpx>=0.27`, `python-dotenv>=1.0`) e `requirements-dev.txt` (`-r requirements.txt`, `pytest>=8.0`, `hypothesis>=6.100`, `reportlab>=4.0` para gerar os PDFs de exemplo)
    - Criar `.gitignore` com `.env`, `data/`, `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`, `.hypothesis/`
    - Criar `.env.example` com todas as variáveis do design (`PROVEDOR_EMBEDDINGS`, `PROVEDOR_GERACAO`, `OPENAI_API_KEY=sua-chave-aqui`, `OLLAMA_HOST`, `MODELO_EMBEDDINGS_LOCAL`, `MODELO_GERACAO_LOCAL`, `MODELO_EMBEDDINGS_EXTERNO`, `MODELO_GERACAO_EXTERNO`, `TAMANHO_TRECHO`, `SOBREPOSICAO`, `TOP_K`, `DIRETORIO_INDICE`), sem nenhuma chave real
    - Criar `LICENSE` com o texto integral da licença MIT, com o ano corrente e o nome do titular dos direitos autorais
    - _Requirements: 6.1, 6.2, 9.2, 9.3_

  - [ ] 1.2 Implementar erros, modelos de dados e mensagens
    - Criar `erros.py` com a hierarquia completa do design (`ErroChatbot`, `ErroConfiguracao`, `ErroChaveApi`, `ErroLimiteCaminhos`, `ErroPdf`, `ErroEmbeddings`, `ErroProvedor`, `ErroIndiceIncompativel`, `ErroIndiceCorrompido`, `ErroPergunta`, `ErroProvedorLocalIndisponivel`); `ErroChaveApi` guarda só o nome da variável e `ErroLimiteCaminhos` guarda `recebidos` e `limite`
    - Criar `modelos.py` com `Trecho`, `Citacao`, `ResultadoBusca`, `ResumoIngestao`, `FalhaArquivo`, `Interacao` e `RespostaChatbot` (dataclasses imutáveis)
    - Criar `mensagens.py` com as mensagens ao usuário em português (pergunta vazia, limite de 2.000 caracteres, índice vazio, não encontrada, não fundamentada, falha de comunicação, nenhum documento indexado e aviso de terceiros)
    - _Requirements: 1.3, 1.7, 1.13, 2.8, 2.11, 3.4, 3.6, 3.7, 4.5, 4.7_

  - [ ] 1.3 Criar os dublês de teste compartilhados
    - Criar `tests/dobles.py` com `EmbeddingsFalso` (determinístico por `hashlib.sha256`, dimensão fixa, falha injetável e contador de chamadas), `GeracaoFalsa` (respostas programadas, sentinela, erro e timeout simulados, registro dos prompts) e `LeitorPdfFalso` (páginas geradas, páginas vazias ou `ErroPdf`)
    - Criar `tests/conftest.py` com os fixtures `tmp_path`-based para índice e diretórios, e o perfil padrão do Hypothesis (`max_examples=100`)
    - _Requirements: 9.5_

- [ ] 2. Implementar a configuração
  - [ ] 2.1 Implementar `config.py`
    - Criar `Configuracao` (dataclass imutável, `chave_api` com `repr=False`), `carregar_configuracao`, `validar_parametros_chunking` e `validar_top_k`
    - Ler o `.env` com `dotenv_values` e mesclar com `os.environ`, com o ambiente vencendo em conflito
    - Rejeitar `PROVEDOR_EMBEDDINGS` e `PROVEDOR_GERACAO` ausentes ou fora de `{local, externo}` com `ErroConfiguracao` citando a variável e os valores aceitos
    - Rejeitar chave ausente, vazia ou só com espaços quando algum provedor é `externo`, com `ErroChaveApi` citando `OPENAI_API_KEY`
    - _Requirements: 1.3, 2.7, 5.1, 5.2, 5.4, 5.5_

  - [ ]* 2.2 Escrever teste de propriedade para a validação dos provedores
    - **Property 23: Validação dos provedores configurados**
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 2.3 Escrever teste de propriedade para a precedência do ambiente sobre o `.env`
    - **Property 24: Precedência do ambiente sobre o `.env`**
    - **Validates: Requirements 5.4**

  - [ ]* 2.4 Escrever teste de propriedade para chave de API em branco
    - **Property 25: Chave de API em branco é rejeitada antes de qualquer requisição**
    - **Validates: Requirements 5.5**

- [ ] 3. Implementar o chunking
  - [ ] 3.1 Implementar `dividir_em_trechos` em `chunking.py`
    - Função pura: `strip` do texto, `[]` para texto vazio, janelas de `tamanho` caracteres com passo `tamanho - sobreposicao`, parando quando a janela alcança o fim do texto
    - _Requirements: 1.2_

  - [ ]* 3.2 Escrever teste de propriedade para o chunking
    - **Property 1: Chunking respeita tamanho, sobreposição e preserva o texto**
    - **Validates: Requirements 1.2**

  - [ ]* 3.3 Escrever testes de exemplo para o chunking
    - Texto vazio, texto menor que o tamanho, sobreposição zero, sobreposição máxima (50%) e texto só com espaços
    - _Requirements: 1.2, 9.5_

- [ ] 4. Implementar o Gerador_de_Embeddings e seus provedores
  - [ ] 4.1 Implementar `embeddings/base.py` com `ProvedorEmbeddings` e `GeradorDeEmbeddings`
    - `gerar` divide em lotes (padrão 64), concatena em `np.ndarray` `float32` e lança `ErroEmbeddings` se a quantidade ou a dimensão dos vetores for inconsistente
    - Expor `nome_modelo`
    - _Requirements: 2.1_

  - [ ] 4.2 Implementar `embeddings/ollama.py` e `embeddings/openai.py`
    - `OllamaEmbeddings`: `POST {host}/api/embed` com `{model, input}`
    - `OpenAIEmbeddings`: `POST https://api.openai.com/v1/embeddings` com `Authorization: Bearer`; 401 e 403 viram `ErroChaveApi` (só com o nome da variável)
    - Receber `httpx.Client` por injeção; erro HTTP, JSON malformado e timeout viram `ErroEmbeddings`
    - _Requirements: 2.1, 2.9, 5.3, 6.4_

  - [ ]* 4.3 Escrever teste de propriedade para os embeddings
    - **Property 9: Embeddings têm um vetor por texto e mesma dimensão**
    - **Validates: Requirements 2.1**

  - [ ]* 4.4 Escrever testes dos adaptadores HTTP com `httpx.MockTransport`
    - Resposta válida, 401, 403, 500, JSON malformado e `httpx.ReadTimeout` para os dois adaptadores
    - _Requirements: 2.1, 2.9, 6.4_

- [ ] 5. Implementar o Indice_Vetorial
  - [ ] 5.1 Implementar `indice_vetorial.py`
    - `IndiceVetorial(diretorio, nome_modelo)` com `carregar`, `substituir_fonte`, `buscar`, `listar_fontes`, `esta_vazio` e `__len__`
    - Persistir `vectors.npy` (vetores normalizados, `float32`), `chunks.json` e `meta.json` (`modelo_embeddings`, `dimensao`, `versao_formato`)
    - `substituir_fonte` monta o novo estado completo, grava temporários no mesmo diretório e troca com `os.replace`; só depois atualiza a memória; rejeita vetor nulo com `ErroEmbeddings`; lista vazia de Trechos remove a fonte
    - `buscar` normaliza a consulta, usa produto matricial e `np.argpartition`, ordena de forma decrescente com desempate por posição de inserção e aceita o filtro opcional `fontes`
    - `carregar` lança `ErroIndiceIncompativel` para modelo diferente e `ErroIndiceCorrompido` para arquivos inconsistentes; diretório inexistente começa vazio
    - _Requirements: 1.9, 1.10, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 5.2 Escrever teste de propriedade para a persistência do índice
    - **Property 10: Persistência do índice é uma ida e volta**
    - **Validates: Requirements 2.2, 2.3, 2.4**

  - [ ]* 5.3 Escrever teste de propriedade para a ordenação da busca
    - **Property 11: Busca devolve os mais similares em ordem decrescente**
    - **Validates: Requirements 2.5**

  - [ ]* 5.4 Escrever teste de propriedade para a ida e volta entre indexar e recuperar
    - **Property 12: Ida e volta entre indexar e recuperar**
    - **Validates: Requirements 2.10**

  - [ ]* 5.5 Escrever testes de exemplo do índice
    - Índice vazio, diretório inexistente, arquivos corrompidos (`ErroIndiceCorrompido`), modelo incompatível, remoção de fonte
    - Desempenho: busca em índice de 10.000 Trechos em menos de 2 s e gravação em menos de 5 s
    - _Requirements: 2.3, 2.4, 2.5_

- [ ] 6. Implementar o Ingestor
  - [ ] 6.1 Implementar `ingestor.py` com `LeitorPdf`, `LeitorPdfPypdf` e `Ingestor`
    - Validar `tamanho` e `sobreposicao` na construção (`ErroConfiguracao`, nenhum arquivo processado)
    - Rejeitar mais de 50 caminhos com `ErroLimiteCaminhos` antes de abrir qualquer arquivo
    - Expandir diretórios para os arquivos `.pdf` diretos (extensão sem diferenciar caixa, sem recursão) e avisar "nenhum PDF encontrado" quando não houver
    - Para PDF: ler páginas, descartar as vazias após `strip`, dividir cada página e criar `Trecho` com `fonte` igual ao nome do arquivo e `pagina` a partir de 1; sem texto extraível gera aviso e ignora o arquivo
    - Para `.txt`: decodificar como UTF-8, um `Trecho(tipo="texto")` por linha não vazia com `pagina` igual ao número da linha; falha de decodificação ou arquivo sem linhas gera `FalhaArquivo` sem indexar nada
    - Gerar embeddings do lote e chamar `indice.substituir_fonte`; cada arquivo isolado em `try/except`, preservando o estado anterior em caso de falha
    - Devolver `ResumoIngestao` com sucessos, falhas ou ignorados, total de Trechos e mensagens
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 7.4, 7.5_

  - [ ]* 6.2 Escrever teste de propriedade para parâmetros de chunking inválidos
    - **Property 2: Parâmetros de chunking inválidos são rejeitados sem processar arquivos**
    - **Validates: Requirements 1.3**

  - [ ]* 6.3 Escrever teste de propriedade para arquivo e página de origem
    - **Property 3: Trechos carregam arquivo e página de origem em ordem**
    - **Validates: Requirements 1.1, 1.4, 2.2**

  - [ ]* 6.4 Escrever teste de propriedade para o limite de 50 caminhos
    - **Property 4: Mais de 50 caminhos rejeita a solicitação inteira**
    - **Validates: Requirements 1.12, 1.13, 1.14**

  - [ ]* 6.5 Escrever teste de propriedade para a descoberta de PDFs em diretório
    - **Property 5: Descoberta de PDFs em diretório**
    - **Validates: Requirements 1.5**

  - [ ]* 6.6 Escrever teste de propriedade para falhas isoladas e resumo consistente
    - **Property 6: Falhas isoladas e resumo consistente**
    - **Validates: Requirements 1.7, 1.8, 1.11**

  - [ ]* 6.7 Escrever teste de propriedade para a reindexação
    - **Property 7: Reindexar substitui os Trechos do arquivo**
    - **Validates: Requirements 1.9**

  - [ ]* 6.8 Escrever teste de propriedade para falha na substituição
    - **Property 8: Falha na substituição preserva o estado anterior**
    - **Validates: Requirements 1.10**

  - [ ]* 6.9 Escrever teste de propriedade para o Documento_de_Sentencas
    - **Property 27: Documento de sentenças gera um item por linha não vazia**
    - **Validates: Requirements 7.4, 7.5**

  - [ ]* 6.10 Escrever testes de exemplo do Ingestor
    - Diretório sem PDFs, PDF sem texto extraível, arquivo inexistente, exatamente 50 caminhos (aceito) e 51 (rejeitado), e `LeitorPdfPypdf` com um PDF real gerado em `tmp_path`
    - _Requirements: 1.5, 1.6, 1.7, 1.8, 1.12_

- [ ] 7. Checkpoint - Ingestão e indexação
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implementar o Recuperador
  - [ ] 8.1 Implementar `recuperador.py`
    - `Recuperador(embeddings, indice, top_k=4)` com `top_k`, `definir_top_k` (usa `validar_top_k`; em erro mantém o valor anterior) e `recuperar`
    - `fontes_mencionadas(pergunta, fontes)`: nome com ou sem extensão, sem diferenciar caixa
    - Com 2 ou mais arquivos mencionados, buscar por arquivo (`fontes=[f]`) com `max(1, 10 // n)` Trechos cada, juntar, ordenar por similaridade e limitar a 10 no total
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.9, 3.5_

  - [ ]* 8.2 Escrever teste de propriedade para a validação de Top_K
    - **Property 13: Validação de Top_K**
    - **Validates: Requirements 2.6, 2.7**

  - [ ]* 8.3 Escrever teste de propriedade para perguntas comparativas
    - **Property 19: Perguntas comparativas recuperam Trechos de cada arquivo mencionado**
    - **Validates: Requirements 3.5**

  - [ ]* 8.4 Escrever testes de exemplo do Recuperador
    - Sucesso com índice populado, índice com menos de `top_k` Trechos, falha no embedding da pergunta (`ErroEmbeddings` sem alterar o índice) e nome de arquivo com caixa diferente na pergunta
    - _Requirements: 2.5, 2.9, 3.5, 9.5_

- [ ] 9. Implementar o Gerador_de_Respostas e seus provedores
  - [ ] 9.1 Implementar `geracao/base.py` e `geracao/prompt.py`
    - `montar_prompt` (função pura): mensagem de sistema (português, apenas os Trechos fornecidos, marcadores `[n]`, citações literais entre aspas no idioma original, sentinela `SEM_RESPOSTA`, uma citação por arquivo comparado), as 5 interações mais recentes e a mensagem final com Trechos numerados `[n] (arquivo, p. N): texto` e a pergunta
    - `GeradorDeRespostas.gerar` chama `provedor.completar` e aplica `limitar_resposta(texto, 2000)` (corta no último espaço antes do limite, sem passar de 2.000 caracteres)
    - _Requirements: 3.1, 3.2, 3.5, 4.3_

  - [ ] 9.2 Implementar `geracao/ollama.py` e `geracao/openai.py`
    - `OllamaGeracao`: `POST {host}/api/chat` com `stream: false`; `OpenAIGeracao`: `POST /v1/chat/completions`
    - Timeout de 60 s repassado ao `httpx`; `httpx.TimeoutException`, erro HTTP e resposta malformada viram `ErroProvedor`, sem devolver resposta parcial; 401 e 403 do provedor externo viram `ErroChaveApi`
    - _Requirements: 3.6, 6.4_

  - [ ]* 9.3 Escrever teste de propriedade para resposta limitada e prompt completo
    - **Property 16: Resposta limitada e prompt completo**
    - **Validates: Requirements 3.1**

  - [ ]* 9.4 Escrever testes de exemplo do Gerador_de_Respostas
    - O prompt exige português e aspas em citações literais, o sentinela é instruído, e o histórico entra em ordem cronológica
    - _Requirements: 3.2, 4.3_

  - [ ]* 9.5 Escrever testes dos adaptadores de geração com `httpx.MockTransport`
    - Resposta válida, 401, 403, 500, JSON malformado e `httpx.ReadTimeout` para `OllamaGeracao` e `OpenAIGeracao`, verificando que nenhum texto parcial é devolvido
    - _Requirements: 3.6, 6.4_

- [ ] 10. Implementar as Citações
  - [ ] 10.1 Implementar `citacoes.py`
    - `extrair_citacoes`: marcadores `\[(\d+)\]`, ignora números fora de `1..len(trechos)`, remove repetições por `(fonte, pagina)` e preserva a ordem da primeira ocorrência
    - `formatar_citacao` (`arquivo.pdf, p. 12` e `sentencas.txt, linha 3`) e `formatar_lista_citacoes` (título "Fontes:", uma por linha)
    - _Requirements: 3.3_

  - [ ]* 10.2 Escrever teste de propriedade para as Citações
    - **Property 17: Citações só dos Trechos usados, sem repetição**
    - **Validates: Requirements 3.3**

  - [ ]* 10.3 Escrever testes de exemplo das Citações
    - Formatação de PDF e de texto, resposta sem marcadores, marcador fora da faixa, marcadores repetidos e lista vazia
    - _Requirements: 3.3, 9.5_

- [ ] 11. Implementar o Chatbot
  - [ ] 11.1 Implementar `chatbot.py`
    - `normalizar_pergunta`, `validar_pergunta` (`ErroPergunta` para vazia ou acima de 2.000 caracteres após `strip`) e `reconhecer_comando` (`sair`, `documentos`, `carregar`, `topk`, sem diferenciar caixa e ignorando espaços nas pontas)
    - `Chatbot.responder` seguindo o algoritmo do design: validar, tratar índice vazio, recuperar, tratar 0 Trechos, gerar, tratar sentinela, extrair Citações, tratar resposta sem Citação, reforço único para perguntas comparativas com arquivo sem Citação e aviso dos arquivos sem citação, e registro no histórico só após sucesso
    - Manter Trechos e Citações apenas em variáveis locais; qualquer `ErroProvedor`, `ErroEmbeddings` ou `ErroChaveApi` propaga sem alterar o histórico
    - `historico` somente leitura, `limpar_historico` e janela de 5 interações (`JANELA_HISTORICO`)
    - _Requirements: 2.8, 2.9, 2.11, 2.12, 2.13, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.2, 4.3, 4.5, 4.6, 4.7, 4.8_

  - [ ]* 11.2 Escrever teste de propriedade para a validação de pergunta
    - **Property 14: Validação de pergunta sem efeitos colaterais**
    - **Validates: Requirements 2.4, 2.11, 2.12, 2.13, 4.5**

  - [ ]* 11.3 Escrever teste de propriedade para índice vazio
    - **Property 15: Índice vazio não aciona recuperação nem modelo**
    - **Validates: Requirements 2.8**

  - [ ]* 11.4 Escrever teste de propriedade para resposta sem fundamento
    - **Property 18: Resposta sem fundamento não é exibida**
    - **Validates: Requirements 3.4, 3.7**

  - [ ]* 11.5 Escrever teste de propriedade para falha do provedor
    - **Property 20: Falha do provedor não deixa rastros**
    - **Validates: Requirements 3.6, 3.8, 3.9, 4.8, 2.9**

  - [ ]* 11.6 Escrever teste de propriedade para a janela de histórico
    - **Property 21: Janela de histórico**
    - **Validates: Requirements 4.2, 4.3**

  - [ ]* 11.7 Escrever testes de exemplo do Chatbot
    - Fluxo feliz com Citações, pergunta comparativa com reforço e aviso de arquivo sem citação, pergunta de exatamente 2.000 e de 2.001 caracteres, e resposta de 2.000 caracteres
    - _Requirements: 2.11, 3.1, 3.3, 3.5, 3.7_

- [ ] 12. Checkpoint - Consulta e conversa
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Implementar o log seguro e a fábrica de provedores
  - [ ] 13.1 Implementar `log_seguro.py`
    - `FiltroSegredos(segredos)` como `logging.Filter` que substitui qualquer segredo por `***` na mensagem formatada, nos argumentos e no texto de exceção
    - Função para instalar o filtro em todos os handlers de log
    - _Requirements: 6.3_

  - [ ] 13.2 Implementar `fabrica.py`
    - Criar `GeradorDeEmbeddings` e `GeradorDeRespostas` a partir de `Configuracao`, escolhendo Ollama ou OpenAI por provedor, com `httpx.Client` e timeouts da configuração (10 s de conexão local e 60 s de geração)
    - Criar `Ingestor`, `IndiceVetorial`, `Recuperador` e `Chatbot` compartilhando o mesmo `GeradorDeEmbeddings`
    - _Requirements: 2.4, 5.1, 5.3_

  - [ ]* 13.3 Escrever testes de exemplo do filtro de segredos
    - Segredo na mensagem, nos argumentos e no texto de exceção, e múltiplos segredos
    - _Requirements: 6.3_

- [ ] 14. Implementar a CLI
  - [ ] 14.1 Implementar `cli.py` e `__main__.py`
    - Subcomandos `chat` (padrão) e `carregar <caminhos...>`; comandos de sessão `carregar` (com `shlex.split`), `documentos`, `topk <n>` e `sair`; qualquer outra entrada vira pergunta
    - `sair` descarta o histórico e termina com código 0; `documentos` lista os arquivos em ordem alfabética, um por linha, ou informa que nenhum documento está indexado
    - Sequência de inicialização: carregar configuração (erro termina com código 1, sem requisição), instalar `FiltroSegredos`, exibir o aviso de terceiros (identificando a OpenAI) antes de qualquer requisição externa, `verificar_disponibilidade_local` (`GET /api/tags`, timeout de 10 s, conferindo o modelo de embeddings e o de geração separadamente, sem fallback para o provedor externo) e criação dos componentes via fábrica
    - O laço captura `ErroChatbot` e mostra a mensagem; `ErroChaveApi` mantém a sessão ativa; exceções inesperadas vão ao log seguro e viram erro genérico; imprimir as três contagens do resumo de ingestão
    - _Requirements: 1.11, 4.1, 4.4, 4.6, 4.7, 4.8, 5.1, 5.2, 5.5, 5.6, 5.7, 5.8, 6.4, 6.8_

  - [ ]* 14.2 Escrever teste de propriedade para o reconhecimento de comandos
    - **Property 22: Reconhecimento de comandos**
    - **Validates: Requirements 4.4, 4.6, 4.7**

  - [ ]* 14.3 Escrever teste de propriedade para a chave de API nas saídas
    - **Property 26: Chave de API nunca aparece em saídas**
    - **Validates: Requirements 5.6, 6.3, 6.4**

  - [ ]* 14.4 Escrever testes de exemplo da CLI
    - Laço interativo com entrada simulada e ordem cronológica das interações, códigos de saída (0 em `sair`, 1 em erro de inicialização) e `topk` inválido mantendo o valor anterior
    - _Requirements: 4.1, 4.4, 2.7_

  - [ ]* 14.5 Escrever testes de segurança e provedores
    - Erro 401 após a inicialização mantém a sessão e cita o nome da variável sem o valor
    - Provedor local não acessa hosts externos (registro de hosts no `MockTransport`)
    - Falha de componente local (embeddings ou geração) encerra com código diferente de zero, indica o componente e o modelo esperado, e não recorre ao provedor externo
    - _Requirements: 5.3, 5.7, 5.8, 6.4, 6.8_

- [ ] 15. Preparar material de exemplo, README e verificação de entrega
  - [ ] 15.1 Criar `inputs/sentencas.txt`
    - UTF-8, de 5 a 100 sentenças, uma por linha, cada uma com no máximo 500 caracteres, em português e com temas que permitam extrair insights (por exemplo, testes, arquitetura e qualidade de software)
    - _Requirements: 7.1, 7.2_

  - [ ] 15.2 Criar o script gerador de PDFs de exemplo e os PDFs
    - Criar `scripts/gerar_pdfs_exemplo.py` com `reportlab`, produzindo de 2 a 4 PDFs de autoria própria (conteúdo original curto sobre RAG, testes de software e arquitetura), com mais de uma página e texto extraível, de até 10 MB cada
    - Executar o script e salvar os PDFs em `inputs/`, licenciados sob a mesma licença MIT do projeto
    - _Requirements: 7.3_

  - [ ] 15.3 Escrever o `README.md` em português
    - Descrição do projeto (100 a 500 palavras) com o cenário do estudante de Engenharia de Software escrevendo o TCC
    - Instalação e execução: pré-requisitos, comandos ordenados e duas subseções (modo local gratuito com Ollama e modo com API externa, listando as variáveis sem credenciais reais)
    - Seção de prints com no mínimo 3 imagens PNG em `docs/img/` por caminho relativo e texto alternativo não vazio; criar `docs/img/.gitkeep`
    - Processo de construção em subseções ordenadas (ingestão, indexação, recuperação, geração), cada uma com no mínimo 2 frases
    - Insights obtidos com a análise das sentenças (no mínimo 3, com título e descrição), limitações conhecidas (no mínimo 3), possíveis evoluções (no mínimo 3, incluindo a troca do índice por ChromaDB ou FAISS)
    - Uso responsável (documentos sensíveis e envio a provedores externos), guarda e revogação de chaves de API e aviso de que respostas de IA podem conter erros e devem ser conferidas nos artigos originais
    - Tabela com nome, origem e licença de cada arquivo de `inputs/`
    - _Requirements: 6.5, 6.6, 6.7, 7.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [ ]* 15.4 Escrever `tests/test_entrega.py` (testes de fumaça de entrega)
    - `.gitignore` cobre `.env`, o índice persistido e ambientes virtuais; `.env.example` contém todas as variáveis e nenhuma chave real
    - `requirements*.txt` com cada dependência em `==` ou `>=`; `LICENSE` OSI com ano e titular
    - `inputs/` com 5 a 100 sentenças válidas e de 2 a 10 PDFs de até 10 MB legíveis pelo `Ingestor` com ao menos uma página com texto
    - Seções e contagens do README, texto alternativo das imagens e existência dos arquivos referenciados (reportando o caminho inexistente)
    - Varredura do termo proibido definido no Requisito 9.4 em documentação, comentários, código e testes
    - _Requirements: 6.1, 6.2, 7.1, 7.2, 7.3, 8.4, 8.9, 9.2, 9.3, 9.4_

- [ ] 16. Checkpoint final - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Subtarefas marcadas com `*` são opcionais (testes) e podem ser puladas para um MVP mais rápido.
- Cada propriedade do design é uma subtarefa própria, anotada com o número da propriedade e os requisitos que ela valida; cada teste de propriedade usa `@settings(max_examples=100)` ou mais e o comentário `# Feature: pdf-rag-chatbot, Property N: <texto>`.
- Os testes de exemplo cobrem, por função, ao menos um caso de sucesso e um de borda ou entrada inválida (`dividir_em_trechos`, `Recuperador.recuperar`, `formatar_citacao` e `extrair_citacoes`), conforme o Requisito 9.5.
- **Git:** todo o trabalho acontece em uma branch de trabalho (por exemplo, `feature/<descricao-curta>`), com `git add` de arquivos específicos e sem `--no-verify`. Commits e push direto na `main` só são permitidos se o remoto for `github.com/Gtvnv/*` (projeto novo e pessoal); em qualquer outro remoto a `main` não recebe commits e as mudanças chegam por Pull Request. `.env`, chaves e o diretório `data/` nunca são commitados.
- **Fora do escopo do agente de código** (verificação manual do design): capturar os 3 ou mais prints reais em `docs/img/` com o provedor local (o teste de entrega 15.4 só passa por completo depois que as imagens existirem), criar o repositório Git novo e conferir com `git log` que não há histórico herdado nem o termo proibido nas mensagens de commit (Requisitos 8.4 e 9.1), e revisar o conteúdo qualitativo de insights e limitações.
- O texto do README e dos demais arquivos do repositório não deve conter o termo proibido pelo Requisito 9.4.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1", "3.1", "4.1", "10.1", "13.1", "15.1"] },
    { "id": 3, "tasks": ["2.2", "3.2", "4.2", "4.3", "5.1", "9.1", "10.2", "13.3", "15.2"] },
    { "id": 4, "tasks": ["2.3", "3.3", "4.4", "5.2", "6.1", "8.1", "9.2", "9.3", "10.3"] },
    { "id": 5, "tasks": ["2.4", "5.3", "6.2", "8.2", "9.4", "11.1", "13.2"] },
    { "id": 6, "tasks": ["5.4", "6.3", "8.3", "9.5", "11.2", "14.1"] },
    { "id": 7, "tasks": ["5.5", "6.4", "8.4", "11.3", "14.2", "15.3"] },
    { "id": 8, "tasks": ["6.5", "11.4", "14.3", "15.4"] },
    { "id": 9, "tasks": ["6.6", "11.5", "14.4", "14.5"] },
    { "id": 10, "tasks": ["6.7", "11.6"] },
    { "id": 11, "tasks": ["6.8", "11.7"] },
    { "id": 12, "tasks": ["6.9"] },
    { "id": 13, "tasks": ["6.10"] }
  ]
}
```
