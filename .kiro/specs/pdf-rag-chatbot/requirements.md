# Requirements Document

## Introduction

O PDF RAG Chatbot é um chatbot interativo que responde perguntas com base no conteúdo de arquivos PDF fornecidos pelo usuário. O sistema usa RAG (Retrieval-Augmented Generation): extrai o texto dos PDFs, divide em trechos, gera embeddings, armazena em um índice vetorial, recupera os trechos mais relevantes para cada pergunta e usa um modelo de IA generativa para redigir a resposta fundamentada nesses trechos, citando arquivo e página.

O cenário de uso é o de um estudante de Engenharia de Software que escreve o TCC e precisa revisar e correlacionar vários artigos científicos. O projeto é pessoal e educacional, publicado como portfólio em um repositório novo no GitHub. A escolha da stack (linguagem, modelos, banco vetorial) fica para a fase de design. A documentação e as mensagens ao usuário são em português.

## Glossary

- **Chatbot**: Aplicação interativa que recebe perguntas do usuário e devolve respostas geradas pelo Gerador_de_Respostas.
- **Ingestor**: Componente que lê arquivos PDF, extrai o texto e o divide em trechos.
- **Trecho**: Fragmento de texto extraído de um PDF, associado ao nome do arquivo e ao número da página de origem.
- **Gerador_de_Embeddings**: Componente que converte Trechos e perguntas em vetores numéricos (embeddings).
- **Indice_Vetorial**: Armazenamento de embeddings de Trechos que permite busca por similaridade.
- **Recuperador**: Componente que consulta o Indice_Vetorial e devolve os Trechos mais similares a uma pergunta.
- **Gerador_de_Respostas**: Componente que usa um modelo de IA generativa para produzir uma resposta a partir da pergunta e dos Trechos recuperados.
- **Citacao**: Referência ao nome do arquivo PDF e ao número da página de um Trecho usado na resposta.
- **Pasta_Inputs**: Diretório `inputs` do repositório, que contém o documento de texto de sentenças e os PDFs de exemplo.
- **Documento_de_Sentencas**: Arquivo de texto na Pasta_Inputs com algumas sentenças usadas para demonstrar a análise pela IA.
- **Arquivo_de_Ambiente**: Arquivo `.env` local que guarda chaves de API e configurações sensíveis, mantido fora do controle de versão.
- **Provedor_Externo**: Serviço de API de terceiros (nuvem) usado para embeddings ou geração de respostas.
- **Provedor_Local**: Modelo executado na máquina do usuário, sem envio de dados a terceiros e sem custo de API.
- **Readme**: Arquivo `README.md` na raiz do repositório com a documentação do projeto.
- **Repositorio**: Repositório Git do projeto no GitHub.
- **Top_K**: Número de Trechos que o Recuperador devolve por pergunta.

## Requirements

### Requirement 1: Carregar arquivos PDF

**User Story:** Como estudante, quero carregar arquivos PDF no sistema, para que o Chatbot possa usar o conteúdo dos meus artigos científicos.

#### Acceptance Criteria

1. WHEN o usuário informa um ou mais caminhos de arquivos PDF válidos (até 50 caminhos por solicitação), THE Ingestor SHALL extrair o texto de cada página de cada arquivo, preservando a ordem das páginas do documento.
2. WHEN o Ingestor extrai o texto de um PDF, THE Ingestor SHALL dividir o texto em Trechos com tamanho máximo configurável entre 100 e 10.000 caracteres (padrão 1.000) e sobreposição configurável entre 0 e 50% do tamanho máximo (padrão 200 caracteres) entre Trechos consecutivos.
3. IF o tamanho máximo ou a sobreposição configurados estão fora dos limites definidos no critério 2, THEN THE Ingestor SHALL rejeitar a configuração, informar ao usuário qual parâmetro é inválido e quais são os limites aceitos, e SHALL não processar nenhum arquivo.
4. THE Ingestor SHALL associar a cada Trecho o nome do arquivo de origem e o número da página de origem (iniciando em 1).
5. WHEN o usuário informa um diretório, THE Ingestor SHALL carregar todos os arquivos com extensão `.pdf` (sem diferenciar maiúsculas de minúsculas) contidos diretamente nesse diretório, sem percorrer subdiretórios.
6. IF o diretório informado não contém nenhum arquivo `.pdf`, THEN THE Ingestor SHALL informar ao usuário que nenhum PDF foi encontrado e SHALL não alterar o Indice_Vetorial.
7. IF o arquivo informado não existe ou não é um PDF legível, THEN THE Ingestor SHALL informar ao usuário o nome do arquivo e o motivo da falha, SHALL não alterar o Indice_Vetorial para esse arquivo e SHALL continuar o processamento dos demais arquivos.
8. IF um PDF não contém texto extraível em nenhuma página, THEN THE Ingestor SHALL avisar o usuário, com o nome do arquivo, que o arquivo não possui texto extraível e SHALL ignorar esse arquivo sem alterar o Indice_Vetorial.
9. WHEN o usuário carrega um PDF já indexado (identificado pelo nome do arquivo), THE Ingestor SHALL substituir os Trechos anteriores desse arquivo, de modo que o Indice_Vetorial contenha exatamente um conjunto de Trechos para esse arquivo.
10. IF a extração ou a substituição dos Trechos de um PDF já indexado falha, THEN THE Ingestor SHALL manter os Trechos anteriores desse arquivo no Indice_Vetorial e informar ao usuário o nome do arquivo e o motivo da falha.
11. WHEN o Ingestor conclui o processamento de uma solicitação, THE Ingestor SHALL informar ao usuário a quantidade de arquivos carregados com sucesso, a quantidade de arquivos com falha ou ignorados e a quantidade total de Trechos indexados.
12. IF o usuário informa mais de 50 caminhos em uma única solicitação (por exemplo, 51), THEN THE Ingestor SHALL rejeitar a solicitação inteira sem processar nenhum arquivo.
13. IF o usuário informa mais de 50 caminhos em uma única solicitação, THEN THE Ingestor SHALL informar ao usuário o limite de 50 caminhos e a quantidade de caminhos recebida.
14. IF o usuário informa mais de 50 caminhos em uma única solicitação, THEN THE Ingestor SHALL manter o Indice_Vetorial inalterado.

### Requirement 2: Indexar e recuperar com busca vetorial

**User Story:** Como estudante, quero que o conteúdo dos PDFs seja indexado com busca vetorial, para que o sistema encontre os trechos relevantes para cada pergunta.

#### Acceptance Criteria

1. WHEN o Ingestor produz Trechos, THE Gerador_de_Embeddings SHALL gerar exatamente um embedding para cada Trecho, todos com a mesma dimensionalidade e produzidos pelo mesmo modelo de embedding.
2. THE Indice_Vetorial SHALL armazenar o embedding de cada Trecho junto com o texto, o nome do arquivo e o número da página (inteiro maior ou igual a 1).
3. WHEN o Indice_Vetorial recebe novos Trechos, THE Indice_Vetorial SHALL persistir os dados em disco em até 5 segundos após a conclusão da indexação, de modo que, ao reiniciar o sistema, o conteúdo indexado esteja disponível sem reindexar os PDFs.
4. WHEN o usuário envia uma pergunta contendo de 1 a 2.000 caracteres após remover espaços em branco nas extremidades, THE Recuperador SHALL gerar o embedding da pergunta com o mesmo modelo usado nos Trechos.
5. WHEN o Recuperador recebe o embedding da pergunta, THE Recuperador SHALL devolver os Top_K Trechos de maior similaridade, ordenados da maior para a menor similaridade, ou todos os Trechos indexados se houver menos de Top_K, em até 2 segundos para um índice de até 10.000 Trechos.
6. THE Recuperador SHALL aceitar um valor de Top_K configurável, inteiro entre 1 e 20, com valor padrão de 4.
7. IF o valor de Top_K configurado não é um inteiro entre 1 e 20, THEN THE Recuperador SHALL rejeitar a configuração, SHALL manter o valor anterior de Top_K e SHALL informar um erro indicando a faixa permitida.
8. IF o Indice_Vetorial está vazio quando o usuário envia uma pergunta, THEN THE Chatbot SHALL informar ao usuário que nenhum PDF foi carregado, SHALL orientar como carregar arquivos e SHALL NOT invocar o Recuperador nem o modelo de linguagem.
9. IF a geração do embedding da pergunta falha, THEN THE Chatbot SHALL informar ao usuário um erro indicando que a busca não pôde ser realizada e SHALL manter o Indice_Vetorial inalterado.
10. FOR ALL Trechos indexados, consultar o Indice_Vetorial com o texto exato de um Trecho SHALL devolver esse Trecho entre os resultados, na primeira posição, com Top_K igual ao padrão (propriedade de ida e volta entre indexar e recuperar).
11. IF a pergunta contém mais de 2.000 caracteres após remover espaços em branco nas extremidades, THEN THE Chatbot SHALL rejeitar a pergunta e exibir um erro de limite de caracteres que informe o limite de 2.000 caracteres.
12. IF a pergunta contém mais de 2.000 caracteres após remover espaços em branco nas extremidades, THEN THE Chatbot SHALL não acionar o Recuperador.
13. IF a pergunta contém mais de 2.000 caracteres após remover espaços em branco nas extremidades, THEN THE Chatbot SHALL não adicionar a entrada ao histórico.

### Requirement 3: Gerar respostas fundamentadas com citações

**User Story:** Como estudante, quero receber respostas geradas por IA com base nos documentos carregados, para que eu possa revisar e correlacionar artigos com rapidez.

#### Acceptance Criteria

1. WHEN o Recuperador devolve de 1 a 10 Trechos para uma pergunta, THE Gerador_de_Respostas SHALL gerar a resposta usando a pergunta e todos os Trechos recuperados como contexto, e SHALL limitar a resposta a no máximo 2.000 caracteres.
2. THE Gerador_de_Respostas SHALL redigir a resposta em português, exceto trechos citados literalmente em outro idioma, que SHALL aparecer entre aspas e no idioma original.
3. WHEN o Gerador_de_Respostas produz uma resposta, THE Chatbot SHALL exibir, junto à resposta, a lista de Citacoes dos Trechos efetivamente usados na resposta, sem repetições, com cada Citacao identificando o arquivo de origem e a localização do Trecho no arquivo.
4. IF o Recuperador devolve 0 Trechos ou os Trechos recuperados não contêm informação suficiente para responder à pergunta, THEN THE Chatbot SHALL exibir uma mensagem informando que a resposta não foi encontrada nos documentos carregados, SHALL NOT exibir lista de Citacoes e SHALL manter a sessão ativa para novas perguntas.
5. WHEN o usuário pergunta sobre a relação entre dois ou mais documentos, THE Recuperador SHALL buscar Trechos de pelo menos 2 arquivos distintos mencionados na pergunta, e THE Gerador_de_Respostas SHALL indicar pelo menos uma Citacao para cada arquivo comparado na resposta.
6. IF o Provedor_Externo ou o Provedor_Local retorna erro ou não responde em 60 segundos, THEN THE Chatbot SHALL interromper a requisição, exibir uma mensagem de erro indicando a falha de comunicação com o provedor, SHALL NOT exibir resposta parcial ou Citacoes, SHALL preservar o histórico da conversa e SHALL manter a sessão ativa para novas perguntas.
7. IF a resposta gerada para uma pergunta com Trechos recuperados não contém nenhuma Citacao, THEN THE Chatbot SHALL exibir uma mensagem informando que a resposta não pôde ser fundamentada nos documentos carregados, em vez de exibir a resposta sem Citacoes.
8. IF o Provedor_Externo ou o Provedor_Local retorna erro ou não responde em 60 segundos, THEN THE Chatbot SHALL descartar completamente os Trechos recuperados e qualquer Citacao associada à pergunta.
9. IF o Provedor_Externo ou o Provedor_Local retorna erro ou não responde em 60 segundos, THEN THE Chatbot SHALL não reter Trechos nem Citacoes para reutilização em uma nova tentativa da mesma pergunta.

### Requirement 4: Conversar de forma interativa

**User Story:** Como estudante, quero conversar com o Chatbot de forma interativa, para que eu possa fazer perguntas de acompanhamento sobre os artigos.

#### Acceptance Criteria

1. WHEN a sessão de conversa é iniciada, THE Chatbot SHALL exibir um prompt de entrada, aceitar uma pergunta digitada pelo usuário e exibir a resposta correspondente antes de aceitar a pergunta seguinte, mantendo a ordem cronológica das interações.
2. WHILE uma sessão de conversa está ativa, THE Chatbot SHALL manter em memória o histórico completo das perguntas e respostas da sessão, até o encerramento da sessão.
3. WHEN o usuário envia uma pergunta de acompanhamento, THE Chatbot SHALL incluir no contexto enviado ao Gerador_de_Respostas as 5 interações (pares pergunta-resposta) mais recentes da sessão, em ordem cronológica, ou todas as interações existentes se houver menos de 5.
4. WHEN o usuário digita o comando de saída "sair" (sem diferenciar maiúsculas de minúsculas e ignorando espaços no início e no fim), THE Chatbot SHALL encerrar a sessão, descartar o histórico da sessão e finalizar o processo com código de saída 0.
5. IF o usuário envia uma pergunta vazia ou composta apenas por espaços em branco, THEN THE Chatbot SHALL exibir uma mensagem indicando que a pergunta não pode ser vazia e solicitar uma nova pergunta, sem acionar o Recuperador e sem adicionar a entrada ao histórico.
6. WHEN o usuário digita o comando "documentos" (sem diferenciar maiúsculas de minúsculas e ignorando espaços no início e no fim), THE Chatbot SHALL exibir o nome de cada arquivo PDF presente no Indice_Vetorial, um por linha, em ordem alfabética, sem acionar o Recuperador e sem adicionar a entrada ao histórico.
7. IF o Indice_Vetorial não contém nenhum arquivo PDF quando o usuário solicita a lista de documentos, THEN THE Chatbot SHALL exibir uma mensagem indicando que nenhum documento está indexado e manter a sessão ativa.
8. IF o Recuperador ou o Gerador_de_Respostas falha ao processar uma pergunta, THEN THE Chatbot SHALL exibir uma mensagem de erro indicando a falha, manter a sessão ativa, preservar o histórico existente e não registrar a pergunta que falhou no histórico.

### Requirement 5: Escolher entre execução local e API externa

**User Story:** Como estudante, quero escolher entre execução local gratuita e uso de API externa, para que eu controle custo e privacidade dos meus documentos.

#### Acceptance Criteria

1. WHEN o Chatbot é iniciado, THE Chatbot SHALL ler do Arquivo_de_Ambiente ou de variáveis de ambiente o valor que seleciona o provedor de embeddings e o provedor de geração de respostas, sem exigir alteração de código, e SHALL aceitar como valores válidos apenas Provedor_Local ou Provedor_Externo para cada um.
2. IF o valor de seleção de provedor está ausente ou não corresponde a um valor válido, THEN THE Chatbot SHALL encerrar a inicialização antes de processar qualquer PDF, SHALL exibir uma mensagem de erro indicando o nome da variável e os valores aceitos, e SHALL não enviar nenhuma requisição de rede.
3. WHERE o Provedor_Local está configurado para embeddings e para geração de respostas, THE Chatbot SHALL executar ambas as operações sem enviar nenhum conteúdo dos PDFs, nem trechos ou perguntas do usuário, a hosts fora da máquina local (0 requisições a hosts externos durante indexação e consulta).
4. WHERE um Provedor_Externo está configurado, THE Chatbot SHALL ler a chave de API do Arquivo_de_Ambiente ou de variáveis de ambiente, e SHALL usar a variável de ambiente quando o mesmo nome estiver definido nas duas fontes.
5. IF o Provedor_Externo está configurado e a chave de API está ausente ou vazia (0 caracteres, ou apenas espaços), THEN THE Chatbot SHALL exibir uma mensagem de erro com o nome da variável de ambiente esperada, SHALL encerrar a inicialização com código de saída diferente de zero e SHALL não enviar nenhuma requisição ao serviço externo.
6. WHERE um Provedor_Externo está configurado e a chave de API está presente, THE Chatbot SHALL exibir na inicialização, antes de enviar qualquer requisição ao serviço externo, um aviso de que trechos dos documentos serão enviados a um serviço de terceiros, identificando o nome do provedor, e SHALL não exibir o valor da chave de API em nenhuma saída.
7. IF o Provedor_Local está configurado e qualquer componente local (embeddings ou geração de respostas) não responde ou não está instalado em até 10 segundos após a tentativa de conexão, THEN THE Chatbot SHALL encerrar a inicialização completamente com código de saída diferente de zero, mesmo que apenas um dos componentes falhe, e SHALL não recorrer automaticamente a um Provedor_Externo.
8. IF o Provedor_Local está configurado e qualquer componente local não responde ou não está instalado em até 10 segundos após a tentativa de conexão, THEN THE Chatbot SHALL exibir uma mensagem de erro que indique qual componente falhou (embeddings ou geração de respostas), o nome do modelo esperado e as instruções para instalá-lo ou iniciá-lo.

### Requirement 6: Proteger segredos e promover uso responsável

**User Story:** Como mantenedor do projeto, quero proteger segredos e usar o projeto de forma responsável, para que chaves e documentos sensíveis não sejam expostos.

#### Acceptance Criteria

1. THE Repositorio SHALL conter um arquivo `.gitignore` na raiz com entradas que excluam do controle de versão o Arquivo_de_Ambiente, o Indice_Vetorial persistido e os diretórios de ambientes virtuais, de modo que `git status` não liste esses itens como não rastreados após serem criados.
2. THE Repositorio SHALL conter um arquivo `.env.example` com o nome de cada variável de configuração exigida pelo Chatbot, com valor vazio ou valor fictício explícito (por exemplo, `sua-chave-aqui`) para cada chave de API, e sem nenhum valor real de chave.
3. WHEN o Chatbot grava uma entrada de log, THE Chatbot SHALL omitir o valor de qualquer chave de API da entrada, de modo que a busca pelo valor da chave nos logs gerados em uma execução completa (inicialização, indexação, consulta e erro) retorne 0 ocorrências.
4. IF uma chave de API ausente ou inválida for detectada após a inicialização (por exemplo, rejeitada pelo Provedor_Externo durante a indexação ou a consulta), THEN THE Chatbot SHALL exibir uma mensagem de erro que indique o nome da variável ausente ou inválida, sem incluir o valor da chave, e SHALL interromper apenas a operação atual sem enviar novos dados a Provedores_Externos. A chave ausente na inicialização é tratada pelo Requisito 5, critério 5.
5. THE Readme SHALL conter uma seção de uso responsável que oriente o usuário a não enviar documentos sensíveis ou confidenciais a Provedores_Externos sem ciência dos riscos, e que informe que o conteúdo dos documentos indexados e das perguntas é transmitido a esses provedores.
6. THE Readme SHALL conter uma seção que oriente a guardar chaves de API em variáveis de ambiente ou no Arquivo_de_Ambiente fora do controle de versão, e que oriente a revogar e substituir uma chave caso ela seja commitada ou exposta.
7. THE Readme SHALL informar que as respostas geradas por IA podem conter erros e devem ser conferidas nos artigos originais antes do uso no TCC.
8. IF uma chave de API ausente ou inválida for detectada após a inicialização, THEN THE Chatbot SHALL permanecer em execução, sem encerrar o processo, aguardando uma nova configuração da chave.

### Requirement 7: Fornecer material de exemplo na pasta inputs

**User Story:** Como autor do portfólio, quero uma Pasta_Inputs com material de exemplo, para que qualquer pessoa reproduza a demonstração do projeto.

#### Acceptance Criteria

1. THE Repositorio SHALL conter uma Pasta_Inputs no diretório raiz do Repositorio.
2. THE Pasta_Inputs SHALL conter o Documento_de_Sentencas em formato de texto simples com codificação UTF-8, com entre 5 e 100 sentenças, uma sentença por linha, e cada sentença com no máximo 500 caracteres.
3. THE Pasta_Inputs SHALL conter entre 2 e 10 PDFs de exemplo, cada um com tamanho máximo de 10 MB, ao menos 1 página com texto extraível pelo Ingestor, e licença que permita redistribuição.
4. WHEN o Ingestor processa o Documento_de_Sentencas, THE Ingestor SHALL tratar cada sentença não vazia como um item de conteúdo indexável distinto, com o nome do arquivo como fonte.
5. IF o Documento_de_Sentencas não puder ser lido como UTF-8 ou não contiver nenhuma sentença não vazia, THEN THE Ingestor SHALL interromper o processamento desse arquivo, informar ao usuário um erro indicando o motivo e a identificação do arquivo, e não indexar nenhum conteúdo desse arquivo.
6. THE Readme SHALL listar, para cada arquivo de exemplo da Pasta_Inputs, o nome do arquivo, a origem (URL ou autoria) e o nome da licença.

### Requirement 8: Documentar o projeto no Readme

**User Story:** Como autor do portfólio, quero um Readme completo, para que visitantes entendam o projeto, o processo e os aprendizados.

#### Acceptance Criteria

1. THE Readme SHALL conter uma seção de descrição do projeto com os objetivos do Chatbot e o cenário do estudante de Engenharia de Software escrevendo o TCC, com no mínimo 100 e no máximo 500 palavras.
2. THE Readme SHALL conter uma seção de instalação e execução com os pré-requisitos, a lista ordenada de comandos para instalar as dependências e iniciar o Chatbot, e duas subseções distintas: modo local gratuito e modo com API externa.
3. THE Readme SHALL conter, na subseção do modo com API externa, a indicação de quais variáveis de configuração o usuário deve definir, sem incluir valores de credenciais reais.
4. THE Readme SHALL conter no mínimo 3 prints de tela do Chatbot em uso, cada um em formato de imagem (PNG ou JPG) armazenado no repositório, referenciado por caminho relativo e acompanhado de texto alternativo não vazio.
5. THE Readme SHALL conter uma seção de processo de construção que descreva, em subseções separadas e nesta ordem, as etapas de ingestão, indexação, recuperação e geração, com no mínimo 2 frases por etapa.
6. THE Readme SHALL conter uma seção de insights e possibilidades aprendidas após a IA analisar as sentenças do Documento_de_Sentencas, com no mínimo 3 insights, cada um com título e descrição de no mínimo 1 frase.
7. THE Readme SHALL conter uma seção de limitações conhecidas com no mínimo 3 itens e uma seção de possíveis evoluções com no mínimo 3 itens.
8. THE Readme SHALL ser escrito em português.
9. IF qualquer caminho relativo de imagem referenciado no Readme não corresponder a um arquivo existente no repositório, THEN THE Readme SHALL ser considerado incompleto e a verificação de entrega SHALL reportar o caminho inexistente.

### Requirement 9: Manter um repositório limpo e independente

**User Story:** Como autor do portfólio, quero um repositório limpo e independente, para que o projeto seja apresentável no GitHub.

#### Acceptance Criteria

1. THE Repositorio SHALL ser um repositório Git novo, dedicado a este projeto, cujo histórico de commits não contenha commits herdados de outros projetos.
2. THE Repositorio SHALL declarar todas as dependências de execução e de teste do projeto em um arquivo de dependências, onde cada dependência possui versão fixada (igualdade exata) ou limite mínimo (maior ou igual a uma versão informada).
3. THE Repositorio SHALL conter, na raiz, um arquivo de licença cujo texto corresponda integralmente a uma licença aprovada pela Open Source Initiative e que inclua o ano e o nome do titular dos direitos autorais.
4. THE Repositorio SHALL conter zero ocorrências do termo proibido de três letras formado, nesta ordem, pelas letras D, I e O (sem distinção entre maiúsculas e minúsculas, como palavra isolada) em arquivos de documentação, comentários, código e mensagens de commit.
5. THE Repositorio SHALL conter testes automatizados que cubram as funções de divisão de Trechos, de recuperação e de formatação de Citacoes, com no mínimo um teste de caso de sucesso e um teste de caso de borda ou entrada inválida por função.
6. WHEN o conjunto de testes automatizados é executado em um ambiente com as dependências declaradas instaladas, THE Repositorio SHALL concluir a execução com todos os testes aprovados e código de saída 0.
7. IF a execução do conjunto de testes automatizados tiver ao menos um teste reprovado, THEN THE Repositorio SHALL encerrar a execução com código de saída diferente de 0 e identificar cada teste reprovado na saída.
