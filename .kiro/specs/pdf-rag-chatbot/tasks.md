# Implementation Plan: PDF RAG Chatbot (Plano de Implementação)

## Overview

A implementação segue o design em Python 3.11+, como CLI interativa (`python -m pdf_rag_chatbot`), com testes em `pytest` + `hypothesis`. A ordem das tarefas vai do núcleo puro (erros, modelos, configuração, chunking) para os adaptadores (embeddings, índice, provedores de geração), depois para a orquestração (Ingestor, Recuperador, Chatbot), a CLI e, por fim, os artefatos de entrega do repositório (`inputs/`, README, licença). Cada etapa termina integrada à anterior, sem código órfão.

Todo o código, os comentários, os nomes de domínio e as mensagens ao usuário ficam em português, seguindo a nomenclatura do design (`Ingestor`, `Recuperador`, `Trecho`, `Citacao` e assim por diante).

## Tasks

- [x] 1. Preparar a estrutura do projeto e a base de testes
  - [x] 1.1 Criar a estrutura do repositório e os arquivos de configuração
    - Criar `src/pdf_rag_chatbot/` (com `__init__.py`, `__main__.py` mínimo), `tests/`, `inputs/` e `docs/img/`
    - Criar `pyproject.toml` com layout `src`, `pythonpath` do pytest e `testpaths = ["tests"]`
    - Criar `requirements.txt` (`pypdf>=4.0`, `numpy>=1.26`, `httpx>=0.27`, `python-dotenv>=1.0`) e `requirements-dev.txt` (`-r requirements.txt`, `pytest>=8.0`, `hypothesis>=6.100`, `reportlab>=4.0` para gerar os PDFs de exemplo)
    - Criar `.gitignore` com `.env`, `data/`, `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`, `.hypothesis/`
    - Criar `.env.example` com todas as variáveis do design (`PROVEDOR_EMBEDDINGS`, `PROVEDOR_GERACAO`, `OPENAI_API_KEY=sua-chave-aqui`, `OLLAMA_HOST`, `MODELO_EMBEDDINGS_LOCAL`, `MODELO_GERACAO_LOCAL`, `MODELO_EMBEDDINGS_EXTERNO`, `MODELO_GERACAO_EXTERNO`, `TAMANHO_TRECHO`, `SOBREPOSICAO`, `TOP_K`, `DIRETORIO_INDICE`), sem nenhuma chave real
    - Criar `LICENSE` com o texto integral da licença MIT, com o ano corrente e o nome do titular dos direitos autorais
    - _Requirements: 6.1, 6.2, 9.2, 9.3_

  - [x] 1.2 Implementar erros, modelos de dados e mensagens
    - Criar `erros.py` com a hierarquia completa do design (`ErroChatbot`, `ErroConfiguracao`, `ErroChaveApi`, `ErroLimiteCaminhos`, `ErroPdf`, `ErroEmbeddings`, `ErroProvedor`, `ErroIndiceIncompativel`, `ErroIndiceCorrompido`, `ErroPergunta`, `ErroProvedorLocalIndisponivel`); `ErroChaveApi` guarda só o nome da variável e `ErroLimiteCaminhos` guarda `recebidos` e `limite`
    - Criar `modelos.py` com `Trecho`, `Citacao`, `ResultadoBusca`, `ResumoIngestao`, `FalhaArquivo`, `Interacao` e `RespostaChatbot` (dataclasses imutáveis)
    - Criar `mensagens.py` com as mensagens ao usuário em português (pergunta vazia, limite de 2.000 caracteres, índice vazio, não encontrada, não fundamentada, falha de comunicação, nenhum documento indexado e aviso de terceiros)
    - _Requirements: 1.3, 1.7, 1.13, 2.8, 2.11, 3.4, 3.6, 3.7, 4.5, 4.7_

  - [x] 1.3 Criar os dublês de teste compartilhados
    - Criar `tests/dobles.py` com `EmbeddingsFalso` (determinístico por `hashlib.sha256`, dimensão fixa, falha injetável e contador de chamadas), `GeracaoFalsa` (respostas programadas, sentinela, erro e timeout simulados, registro dos prompts) e `LeitorPdfFalso` (páginas geradas, páginas vazias ou `ErroPdf`)
    - Criar `tests/conftest.py` com os fixtures `tmp_path`-based para índice e diretórios, e o perfil padrão do Hypothesis (`max_examples=100`)
    - _Requirements: 9.5_

- [x] 2. Implementar a configuração
  - [x] 2.1 Implementar `config.py`
    - Criar `Configuracao` (dataclass imutável, `chave_api` com `repr=False`), `carregar_configuracao`, `validar_parametros_chunking` e `validar_top_k`
    - Ler o `.env` com `dotenv_values` e mesclar com `os.environ`, com o ambiente vencendo em conflito
    - Rejeitar `PROVEDOR_EMBEDDINGS` e `PROVEDOR_GERACAO` ausentes ou fora de `{local, externo}` com `ErroConfiguracao` citando a variável e os valores aceitos
    - Rejeitar chave ausente, vazia ou só com espaços quando algum provedor é `externo`, com `ErroChaveApi` citando `OPENAI_API_KEY`
    - _Requirements: 1.3, 2.7, 5.1, 5.2, 5.4, 5.5_

  - [x] 2.2 Escrever teste de propriedade para a validação dos provedores
    - **Property 23: Validação dos provedores configurados**
    - **Validates: Requirements 5.1, 5.2**

  - [x] 2.3 Escrever teste de propriedade para a precedência do ambiente sobre o `.env`
    - **Property 24: Precedência do ambiente sobre o `.env`**
    - **Validates: Requirements 5.4**

  - [x] 2.4 Escrever teste de propriedade para chave de API em branco
    - **Property 25: Chave de API em branco é rejeitada antes de qualquer requisição**
    - **Validates: Requirements 5.5**

- [x] 3. Implementar o chunking
  - [x] 3.1 Implementar `dividir_em_trechos` em `chunking.py`
    - Função pura: `strip` do texto, `[]` para texto vazio, janelas de `tamanho` caracteres com passo `tamanho - sobreposicao`, parando quando a janela alcança o fim do texto
    - _Requirements: 1.2_

  - [x] 3.2 Escrever teste de propriedade para o chunking
    - **Property 1: Chunking respeita tamanho, sobreposição e preserva o texto**
    - **Validates: Requirements 1.2**

  - [x] 3.3 Escrever testes de exemplo para o chunking
    - Texto vazio, texto menor que o tamanho, sobreposição zero, sobreposição máxima (50%) e texto só com espaços
    - _Requirements: 1.2, 9.5_

- [x] 4. Implementar o Gerador_de_Embeddings e seus provedores
  - [x] 4.1 Implementar `embeddings/base.py` com `ProvedorEmbeddings` e `GeradorDeEmbeddings`
    - `gerar` divide em lotes (padrão 64), concatena em `np.ndarray` `float32` e lança `ErroEmbeddings` se a quantidade ou a dimensão dos vetores for inconsistente
    - Expor `nome_modelo`
    - _Requirements: 2.1_

  - [x] 4.2 Implementar `embeddings/ollama.py` e `embeddings/openai.py`
    - `OllamaEmbeddings`: `POST {host}/api/embed` com `{model, input}`
    - `OpenAIEmbeddings`: `POST https://api.openai.com/v1/embeddings` com `Authorization: Bearer`; 401 e 403 viram `ErroChaveApi` (só com o nome da variável)
    - Receber `httpx.Client` por injeção; erro HTTP, JSON malformado e timeout viram `ErroEmbeddings`
    - _Requirements: 2.1, 2.9, 5.3, 6.4_

  - [x] 4.3 Escrever teste de propriedade para os embeddings
    - **Property 9: Embeddings têm um vetor por texto e mesma dimensão**
    - **Validates: Requirements 2.1**

  - [x] 4.4 Escrever testes dos adaptadores HTTP com `httpx.MockTransport`
    - Resposta válida, 401, 403, 500, JSON malformado e `httpx.ReadTimeout` para os dois adaptadores
    - _Requirements: 2.1, 2.9, 6.4_

- [x] 5. Implementar o Indice_Vetorial
  - [x] 5.1 Implementar `indice_vetorial.py`
    - `IndiceVetorial(diretorio, nome_modelo)` com `carregar`, `substituir_fonte`, `buscar`, `listar_fontes`, `esta_vazio` e `__len__`
    - Persistir `vectors.npy` (vetores normalizados, `float32`), `chunks.json` e `meta.json` (`modelo_embeddings`, `dimensao`, `versao_formato`)
    - `substituir_fonte` monta o novo estado completo, grava temporários no mesmo diretório e troca com `os.replace`; só depois atualiza a memória; rejeita vetor nulo com `ErroEmbeddings`; lista vazia de Trechos remove a fonte
    - `buscar` normaliza a consulta, usa produto matricial e `np.argpartition`, ordena de forma decrescente com desempate por posição de inserção e aceita o filtro opcional `fontes`
    - `carregar` lança `ErroIndiceIncompativel` para modelo diferente e `ErroIndiceCorrompido` para arquivos inconsistentes; diretório inexistente começa vazio
    - _Requirements: 1.9, 1.10, 2.2, 2.3, 2.4, 2.5_

  - [x] 5.2 Escrever teste de propriedade para a persistência do índice
    - **Property 10: Persistência do índice é uma ida e volta**
    - **Validates: Requirements 2.2, 2.3, 2.4**

  - [x] 5.3 Escrever teste de propriedade para a ordenação da busca
    - **Property 11: Busca devolve os mais similares em ordem decrescente**
    - **Validates: Requirements 2.5**

  - [x] 5.4 Escrever teste de propriedade para a ida e volta entre indexar e recuperar
    - **Property 12: Ida e volta entre indexar e recuperar**
    - **Validates: Requirements 2.10**

  - [x] 5.5 Escrever testes de exemplo do índice
    - Índice vazio, diretório inexistente, arquivos corrompidos (`ErroIndiceCorrompido`), modelo incompatível, remoção de fonte
    - Desempenho: busca em índice de 10.000 Trechos em menos de 2 s e gravação em menos de 5 s
    - _Requirements: 2.3, 2.4, 2.5_

- [x] 6. Implementar o Ingestor
  - [x] 6.1 Implementar `ingestor.py` com `LeitorPdf`, `LeitorPdfPypdf` e `Ingestor`
    - Validar `tamanho` e `sobreposicao` na construção (`ErroConfiguracao`, nenhum arquivo processado)
    - Rejeitar mais de 50 caminhos com `ErroLimiteCaminhos` antes de abrir qualquer arquivo
    - Expandir diretórios para os arquivos `.pdf` diretos (extensão sem diferenciar caixa, sem recursão) e avisar "nenhum PDF encontrado" quando não houver
    - Para PDF: ler páginas, descartar as vazias após `strip`, dividir cada página e criar `Trecho` com `fonte` igual ao nome do arquivo e `pagina` a partir de 1; sem texto extraível gera aviso e ignora o arquivo
    - Para `.txt`: decodificar como UTF-8, um `Trecho(tipo="texto")` por linha não vazia com `pagina` igual ao número da linha; falha de decodificação ou arquivo sem linhas gera `FalhaArquivo` sem indexar nada
    - Gerar embeddings do lote e chamar `indice.substituir_fonte`; cada arquivo isolado em `try/except`, preservando o estado anterior em caso de falha
    - Devolver `ResumoIngestao` com sucessos, falhas ou ignorados, total de Trechos e mensagens
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 7.4, 7.5_

  - [x] 6.2 Escrever teste de propriedade para parâmetros de chunking inválidos
    - **Property 2: Parâmetros de chunking inválidos são rejeitados sem processar arquivos**
    - **Validates: Requirements 1.3**

  - [x] 6.3 Escrever teste de propriedade para arquivo e página de origem
    - **Property 3: Trechos carregam arquivo e página de origem em ordem**
    - **Validates: Requirements 1.1, 1.4, 2.2**

  - [x] 6.4 Escrever teste de propriedade para o limite de 50 caminhos
    - **Property 4: Mais de 50 caminhos rejeita a solicitação inteira**
    - **Validates: Requirements 1.12, 1.13, 1.14**

  - [x] 6.5 Escrever teste de propriedade para a descoberta de PDFs em diretório
    - **Property 5: Descoberta de PDFs em diretório**
    - **Validates: Requirements 1.5**

  - [x] 6.6 Escrever teste de propriedade para falhas isoladas e resumo consistente
    - **Property 6: Falhas isoladas e resumo consistente**
    - **Validates: Requirements 1.7, 1.8, 1.11**

  - [x] 6.7 Escrever teste de propriedade para a reindexação
    - **Property 7: Reindexar substitui os Trechos do arquivo**
    - **Validates: Requirements 1.9**

  - [x] 6.8 Escrever teste de propriedade para falha na substituição
    - **Property 8: Falha na substituição preserva o estado anterior**
    - **Validates: Requirements 1.10**

  - [x] 6.9 Escrever teste de propriedade para o Documento_de_Sentencas
    - **Property 27: Documento de sentenças gera um item por linha não vazia**
    - **Validates: Requirements 7.4, 7.5**

  - [x] 6.10 Escrever testes de exemplo do Ingestor
    - Diretório sem PDFs, PDF sem texto extraível, arquivo inexistente, exatamente 50 caminhos (aceito) e 51 (rejeitado), e `LeitorPdfPypdf` com um PDF real gerado em `tmp_path`
    - _Requirements: 1.5, 1.6, 1.7, 1.8, 1.12_

- [x] 7. Checkpoint - Ingestão e indexação
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implementar o Recuperador
  - [x] 8.1 Implementar `recuperador.py`
    - `Recuperador(embeddings, indice, top_k=4)` com `top_k`, `definir_top_k` (usa `validar_top_k`; em erro mantém o valor anterior) e `recuperar`
    - `fontes_mencionadas(pergunta, fontes)`: nome com ou sem extensão, sem diferenciar caixa
    - Com 2 ou mais arquivos mencionados, buscar por arquivo (`fontes=[f]`) com `max(1, 10 // n)` Trechos cada, juntar, ordenar por similaridade e limitar a 10 no total
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.9, 3.5_

  - [x] 8.2 Escrever teste de propriedade para a validação de Top_K
    - **Property 13: Validação de Top_K**
    - **Validates: Requirements 2.6, 2.7**

  - [x] 8.3 Escrever teste de propriedade para perguntas comparativas
    - **Property 19: Perguntas comparativas recuperam Trechos de cada arquivo mencionado**
    - **Validates: Requirements 3.5**

  - [x] 8.4 Escrever testes de exemplo do Recuperador
    - Sucesso com índice populado, índice com menos de `top_k` Trechos, falha no embedding da pergunta (`ErroEmbeddings` sem alterar o índice) e nome de arquivo com caixa diferente na pergunta
    - _Requirements: 2.5, 2.9, 3.5, 9.5_

- [x] 9. Implementar o Gerador_de_Respostas e seus provedores
  - [x] 9.1 Implementar `geracao/base.py` e `geracao/prompt.py`
    - `montar_prompt` (função pura): mensagem de sistema (português, apenas os Trechos fornecidos, marcadores `[n]`, citações literais entre aspas no idioma original, sentinela `SEM_RESPOSTA`, uma citação por arquivo comparado), as 5 interações mais recentes e a mensagem final com Trechos numerados `[n] (arquivo, p. N): texto` e a pergunta
    - `GeradorDeRespostas.gerar` chama `provedor.completar` e aplica `limitar_resposta(texto, 2000)` (corta no último espaço antes do limite, sem passar de 2.000 caracteres)
    - _Requirements: 3.1, 3.2, 3.5, 4.3_

  - [x] 9.2 Implementar `geracao/ollama.py` e `geracao/openai.py`
    - `OllamaGeracao`: `POST {host}/api/chat` com `stream: false`; `OpenAIGeracao`: `POST /v1/chat/completions`
    - Timeout de 60 s repassado ao `httpx`; `httpx.TimeoutException`, erro HTTP e resposta malformada viram `ErroProvedor`, sem devolver resposta parcial; 401 e 403 do provedor externo viram `ErroChaveApi`
    - _Requirements: 3.6, 6.4_

  - [x] 9.3 Escrever teste de propriedade para resposta limitada e prompt completo
    - **Property 16: Resposta limitada e prompt completo**
    - **Validates: Requirements 3.1**

  - [x] 9.4 Escrever testes de exemplo do Gerador_de_Respostas
    - O prompt exige português e aspas em citações literais, o sentinela é instruído, e o histórico entra em ordem cronológica
    - _Requirements: 3.2, 4.3_

  - [x] 9.5 Escrever testes dos adaptadores de geração com `httpx.MockTransport`
    - Resposta válida, 401, 403, 500, JSON malformado e `httpx.ReadTimeout` para `OllamaGeracao` e `OpenAIGeracao`, verificando que nenhum texto parcial é devolvido
    - _Requirements: 3.6, 6.4_

- [x] 10. Implementar as Citações
  - [x] 10.1 Implementar `citacoes.py`
    - `extrair_citacoes`: marcadores `\[(\d+)\]`, ignora números fora de `1..len(trechos)`, remove repetições por `(fonte, pagina)` e preserva a ordem da primeira ocorrência
    - `formatar_citacao` (`arquivo.pdf, p. 12` e `sentencas.txt, linha 3`) e `formatar_lista_citacoes` (título "Fontes:", uma por linha)
    - _Requirements: 3.3_

  - [x] 10.2 Escrever teste de propriedade para as Citações
    - **Property 17: Citações só dos Trechos usados, sem repetição**
    - **Validates: Requirements 3.3**

  - [x] 10.3 Escrever testes de exemplo das Citações
    - Formatação de PDF e de texto, resposta sem marcadores, marcador fora da faixa, marcadores repetidos e lista vazia
    - _Requirements: 3.3, 9.5_

- [x] 11. Implementar o Chatbot
  - [x] 11.1 Implementar `chatbot.py`
    - `normalizar_pergunta`, `validar_pergunta` (`ErroPergunta` para vazia ou acima de 2.000 caracteres após `strip`) e `reconhecer_comando` (`sair`, `documentos`, `carregar`, `topk`, sem diferenciar caixa e ignorando espaços nas pontas)
    - `Chatbot.responder` seguindo o algoritmo do design: validar, tratar índice vazio, recuperar, tratar 0 Trechos, gerar, tratar sentinela, extrair Citações, tratar resposta sem Citação, reforço único para perguntas comparativas com arquivo sem Citação e aviso dos arquivos sem citação, e registro no histórico só após sucesso
    - Manter Trechos e Citações apenas em variáveis locais; qualquer `ErroProvedor`, `ErroEmbeddings` ou `ErroChaveApi` propaga sem alterar o histórico
    - `historico` somente leitura, `limpar_historico` e janela de 5 interações (`JANELA_HISTORICO`)
    - _Requirements: 2.8, 2.9, 2.11, 2.12, 2.13, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.2, 4.3, 4.5, 4.6, 4.7, 4.8_

  - [x] 11.2 Escrever teste de propriedade para a validação de pergunta
    - **Property 14: Validação de pergunta sem efeitos colaterais**
    - **Validates: Requirements 2.4, 2.11, 2.12, 2.13, 4.5**

  - [x] 11.3 Escrever teste de propriedade para índice vazio
    - **Property 15: Índice vazio não aciona recuperação nem modelo**
    - **Validates: Requirements 2.8**

  - [x] 11.4 Escrever teste de propriedade para resposta sem fundamento
    - **Property 18: Resposta sem fundamento não é exibida**
    - **Validates: Requirements 3.4, 3.7**

  - [x] 11.5 Escrever teste de propriedade para falha do provedor
    - **Property 20: Falha do provedor não deixa rastros**
    - **Validates: Requirements 3.6, 3.8, 3.9, 4.8, 2.9**

  - [x] 11.6 Escrever teste de propriedade para a janela de histórico
    - **Property 21: Janela de histórico**
    - **Validates: Requirements 4.2, 4.3**

  - [x] 11.7 Escrever testes de exemplo do Chatbot
    - Fluxo feliz com Citações, pergunta comparativa com reforço e aviso de arquivo sem citação, pergunta de exatamente 2.000 e de 2.001 caracteres, e resposta de 2.000 caracteres
    - _Requirements: 2.11, 3.1, 3.3, 3.5, 3.7_

- [x] 12. Checkpoint - Consulta e conversa
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implementar o log seguro e a fábrica de provedores
  - [x] 13.1 Implementar `log_seguro.py`
    - `FiltroSegredos(segredos)` como `logging.Filter` que substitui qualquer segredo por `***` na mensagem formatada, nos argumentos e no texto de exceção
    - Função para instalar o filtro em todos os handlers de log
    - _Requirements: 6.3_

  - [x] 13.2 Implementar `fabrica.py`
    - Criar `GeradorDeEmbeddings` e `GeradorDeRespostas` a partir de `Configuracao`, escolhendo Ollama ou OpenAI por provedor, com `httpx.Client` e timeouts da configuração (10 s de conexão local e 60 s de geração)
    - Criar `Ingestor`, `IndiceVetorial`, `Recuperador` e `Chatbot` compartilhando o mesmo `GeradorDeEmbeddings`
    - _Requirements: 2.4, 5.1, 5.3_

  - [x] 13.3 Escrever testes de exemplo do filtro de segredos
    - Segredo na mensagem, nos argumentos e no texto de exceção, e múltiplos segredos
    - _Requirements: 6.3_

- [x] 14. Implementar a CLI
  - [x] 14.1 Implementar `cli.py` e `__main__.py`
    - Subcomandos `chat` (padrão) e `carregar <caminhos...>`; comandos de sessão `carregar` (com `shlex.split`), `documentos`, `topk <n>` e `sair`; qualquer outra entrada vira pergunta
    - `sair` descarta o histórico e termina com código 0; `documentos` lista os arquivos em ordem alfabética, um por linha, ou informa que nenhum documento está indexado
    - Sequência de inicialização: carregar configuração (erro termina com código 1, sem requisição), instalar `FiltroSegredos`, exibir o aviso de terceiros (identificando a OpenAI) antes de qualquer requisição externa, `verificar_disponibilidade_local` (`GET /api/tags`, timeout de 10 s, conferindo o modelo de embeddings e o de geração separadamente, sem fallback para o provedor externo) e criação dos componentes via fábrica
    - O laço captura `ErroChatbot` e mostra a mensagem; `ErroChaveApi` mantém a sessão ativa; exceções inesperadas vão ao log seguro e viram erro genérico; imprimir as três contagens do resumo de ingestão
    - _Requirements: 1.11, 4.1, 4.4, 4.6, 4.7, 4.8, 5.1, 5.2, 5.5, 5.6, 5.7, 5.8, 6.4, 6.8_

  - [x] 14.2 Escrever teste de propriedade para o reconhecimento de comandos
    - **Property 22: Reconhecimento de comandos**
    - **Validates: Requirements 4.4, 4.6, 4.7**

  - [x] 14.3 Escrever teste de propriedade para a chave de API nas saídas
    - **Property 26: Chave de API nunca aparece em saídas**
    - **Validates: Requirements 5.6, 6.3, 6.4**

  - [x] 14.4 Escrever testes de exemplo da CLI
    - Laço interativo com entrada simulada e ordem cronológica das interações, códigos de saída (0 em `sair`, 1 em erro de inicialização) e `topk` inválido mantendo o valor anterior
    - _Requirements: 4.1, 4.4, 2.7_

  - [x] 14.5 Escrever testes de segurança e provedores
    - Erro 401 após a inicialização mantém a sessão e cita o nome da variável sem o valor
    - Provedor local não acessa hosts externos (registro de hosts no `MockTransport`)
    - Falha de componente local (embeddings ou geração) encerra com código diferente de zero, indica o componente e o modelo esperado, e não recorre ao provedor externo
    - _Requirements: 5.3, 5.7, 5.8, 6.4, 6.8_

- [x] 15. Preparar material de exemplo, README e verificação de entrega
  - [x] 15.1 Criar `inputs/sentencas.txt`
    - UTF-8, de 5 a 100 sentenças, uma por linha, cada uma com no máximo 500 caracteres, em português e com temas que permitam extrair insights (por exemplo, testes, arquitetura e qualidade de software)
    - _Requirements: 7.1, 7.2_

  - [x] 15.2 Criar o script gerador de PDFs de exemplo e os PDFs
    - Criar `scripts/gerar_pdfs_exemplo.py` com `reportlab`, produzindo de 2 a 4 PDFs de autoria própria (conteúdo original curto sobre RAG, testes de software e arquitetura), com mais de uma página e texto extraível, de até 10 MB cada
    - Executar o script e salvar os PDFs em `inputs/`, licenciados sob a mesma licença MIT do projeto
    - _Requirements: 7.3_

  - [x] 15.3 Escrever o `README.md` em português
    - Descrição do projeto (100 a 500 palavras) com o cenário do estudante de Engenharia de Software escrevendo o TCC
    - Instalação e execução: pré-requisitos, comandos ordenados e duas subseções (modo local gratuito com Ollama e modo com API externa, listando as variáveis sem credenciais reais)
    - Seção de prints com no mínimo 3 imagens PNG em `docs/img/` por caminho relativo e texto alternativo não vazio; criar `docs/img/.gitkeep`
    - Processo de construção em subseções ordenadas (ingestão, indexação, recuperação, geração), cada uma com no mínimo 2 frases
    - Insights obtidos com a análise das sentenças (no mínimo 3, com título e descrição), limitações conhecidas (no mínimo 3), possíveis evoluções (no mínimo 3, incluindo a troca do índice por ChromaDB ou FAISS)
    - Uso responsável (documentos sensíveis e envio a provedores externos), guarda e revogação de chaves de API e aviso de que respostas de IA podem conter erros e devem ser conferidas nos artigos originais
    - Tabela com nome, origem e licença de cada arquivo de `inputs/`
    - _Requirements: 6.5, 6.6, 6.7, 7.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [x] 15.4 Escrever `tests/test_entrega.py` (testes de fumaça de entrega)
    - `.gitignore` cobre `.env`, o índice persistido e ambientes virtuais; `.env.example` contém todas as variáveis e nenhuma chave real
    - `requirements*.txt` com cada dependência em `==` ou `>=`; `LICENSE` OSI com ano e titular
    - `inputs/` com 5 a 100 sentenças válidas e de 2 a 10 PDFs de até 10 MB legíveis pelo `Ingestor` com ao menos uma página com texto
    - Seções e contagens do README, texto alternativo das imagens e existência dos arquivos referenciados (reportando o caminho inexistente)
    - Varredura do termo proibido definido no Requisito 9.4 em documentação, comentários, código e testes
    - _Requirements: 6.1, 6.2, 7.1, 7.2, 7.3, 8.4, 8.9, 9.2, 9.3, 9.4_

- [x] 16. Checkpoint final - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Tornar configuráveis o tempo limite e a temperatura de geração (Revisão 2)
  - [x] 17.1 Adicionar `tipos.py` e as novas configurações em `config.py`
    - Criar `src/pdf_rag_chatbot/tipos.py` com o alias `TempoLimite = float | httpx.Timeout`
    - `validar_timeout_geracao` (inteiro de 10 a 600) e `validar_temperatura` (número de 0 a 1), lançando `ErroConfiguracao` com o nome da variável e a faixa
    - Ler `TIMEOUT_GERACAO_S` (padrão 60) e `TEMPERATURA_GERACAO` (padrão 0,2) em `carregar_configuracao`; campos `timeout_geracao_s: int` e `temperatura_geracao: float` em `Configuracao`
    - Acrescentar as duas variáveis, com os valores padrão, em `.env.example`
    - _Requirements: 3.10, 3.11, 3.13, 3.14_
  - [x] 17.2 Repassar tempo limite e temperatura aos provedores e corrigir a tipagem
    - `OllamaGeracao` envia `"options": {"temperature": t}` e `OpenAIGeracao` envia `"temperature": t`, recebendo `temperatura` no construtor
    - Usar `TempoLimite` em `timeout_s` dos provedores de embeddings e de geração, no `Protocol` `ProvedorGeracao` e no `GeradorDeRespostas`
    - `fabrica.py` repassa `config.timeout_geracao_s` e `config.temperatura_geracao` e não usa mais `# type: ignore[arg-type]`
    - _Requirements: 3.6, 3.10, 3.13_
  - [x] 17.3 Escrever teste de propriedade para tempo limite e temperatura
    - **Property 28: Tempo limite e temperatura de geração são validados**
    - **Validates: Requirements 3.10, 3.11, 3.13, 3.14**
  - [x] 17.4 Escrever testes de exemplo dos provedores com temperatura e tempo limite
    - Temperatura no corpo das requisições do Ollama e da OpenAI; tempo limite configurado repassado ao `httpx`; valores padrão quando as variáveis estão ausentes
    - _Requirements: 3.6, 3.10, 3.13_
- [x] 18. Tratar resposta só com marcadores como não fundamentada (Revisão 2)
  - [x] 18.1 Implementar `tem_conteudo` em `chatbot.py` e usá-la em `responder`, inclusive na resposta do reforço comparativo
    - _Requirements: 3.7, 3.12_
  - [x] 18.2 Escrever teste de propriedade para resposta só com marcadores
    - **Property 29: Resposta só com marcadores é não fundamentada**
    - **Validates: Requirements 3.12, 3.7**
- [x] 19. Detectar menção de arquivo como termo isolado (Revisão 2)
  - [x] 19.1 Reescrever `fontes_mencionadas` com `(?<![\w])` + `re.escape(nome)` + `(?![\w])` e `re.IGNORECASE`, para o nome com e sem extensão
    - _Requirements: 3.5_
  - [x] 19.2 Escrever teste de propriedade para menção de arquivo
    - **Property 30: Menção de arquivo exige termo isolado**
    - **Validates: Requirements 3.5**
- [x] 20. Tornar atômica a persistência do índice com manifesto (Revisão 2)
  - [x] 20.1 Implementar o formato 2 em `indice_vetorial.py`
    - `versoes/<id>/` com os três arquivos e `atual.json` trocado com um único `os.replace`, com `fsync` antes da troca
    - Limpeza tolerante a falhas das versões não atuais depois da troca; versões órfãs ignoradas na carga
    - Carga do formato 1 (arquivos na raiz) e migração na próxima gravação; manifesto para versão inexistente lança `ErroIndiceCorrompido`
    - Ajustar os testes existentes que dependem do layout antigo (`tests/test_indice.py` e `_estado_em_disco` em `tests/test_ingestor_propriedades.py`)
    - _Requirements: 1.10, 2.3, 2.14_
  - [x] 20.2 Escrever teste de propriedade para gravação interrompida
    - **Property 31: Gravação interrompida do índice preserva um estado completo**
    - **Validates: Requirements 2.14, 1.10**
  - [x] 20.3 Escrever testes de exemplo do formato 2
    - Carga e migração do formato 1, manifesto apontando para versão inexistente, versão órfã ignorada e removida na gravação seguinte
    - _Requirements: 2.3, 2.14_
- [x] 21. Contêineres, integração contínua e documentação (Revisão 2)
  - [x] 21.1 Atualizar `docker-compose.yml`
    - Imagem `ollama/ollama:0.40.16`; serviço `ollama-init` que baixa os modelos e falha com `Falha ao baixar o modelo <nome>`; `chatbot` dependente de `ollama-init` com `service_completed_successfully`
    - Repassar `TIMEOUT_GERACAO_S` e `TEMPERATURA_GERACAO`; manter volumes de modelos e índice e o usuário sem privilégios do `Dockerfile`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  - [x] 21.2 Criar `.github/workflows/testes.yml`
    - `push` e `pull_request` para `main`, matriz Python 3.11 e 3.12, instalação de `requirements-dev.txt` e do pacote, execução do `pytest`
    - _Requirements: 9.6, 9.7, 9.8_
  - [x] 21.3 Atualizar o `README.md`
    - Badge do workflow de CI; instruções do Docker sem download manual de modelos; dica de modelo menor e de `TIMEOUT_GERACAO_S` para máquinas sem GPU; novas variáveis na tabela de parâmetros; manter as seções e formatos exigidos pelo teste de entrega
    - _Requirements: 3.10, 3.13, 9.9, 10.7_
  - [x] 21.4 Estender `tests/test_entrega.py`
    - `Dockerfile` com usuário sem privilégios; compose com imagem fixada, `ollama-init` e `service_completed_successfully`; `.dockerignore`; workflow de CI; badge no README; `.gitattributes` com PDFs e PNGs binários; novas variáveis em `.env.example`
    - _Requirements: 9.8, 9.9, 10.1, 10.2, 10.3, 10.5, 10.7_
- [x] 22. Checkpoint da Revisão 2 - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Executar o Ambiente_de_Conteineres de verdade, recapturar as evidências com `scripts/capturar_evidencias.py` e regenerar as imagens com `scripts/renderizar_evidencias.py`

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
    { "id": 13, "tasks": ["6.10"] },
    { "id": 14, "tasks": ["17.1", "18.1", "19.1", "20.1", "21.1"] },
    { "id": 15, "tasks": ["17.2", "18.2", "19.2", "20.2", "21.2"] },
    { "id": 16, "tasks": ["17.3", "20.3", "21.3"] },
    { "id": 17, "tasks": ["17.4", "21.4"] }
  ]
}
```
