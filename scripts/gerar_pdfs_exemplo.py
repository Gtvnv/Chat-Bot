"""Gera os PDFs de exemplo da pasta ``inputs/``.

Os textos são de autoria própria e seguem a mesma licença MIT do projeto.
Uso: ``python scripts/gerar_pdfs_exemplo.py``
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

DIRETORIO_SAIDA = Path(__file__).resolve().parent.parent / "inputs"
LICENCA = "Conteúdo de autoria própria, licenciado sob a licença MIT (ver LICENSE do projeto)."

# Cada documento: nome do arquivo, título e lista de seções (título, parágrafos).
DOCUMENTOS: list[dict] = [
    {
        "arquivo": "introducao_rag.pdf",
        "titulo": "Introdução à Geração Aumentada por Recuperação (RAG)",
        "secoes": [
            (
                "1. O problema que o RAG resolve",
                [
                    "Modelos de linguagem aprendem a partir de um conjunto fixo de textos. "
                    "Depois do treinamento, eles não conhecem documentos novos nem dados privados, "
                    "e podem inventar fatos com aparência de verdade quando não têm a informação.",
                    "A Geração Aumentada por Recuperação, conhecida pela sigla RAG, reduz esse "
                    "risco. Antes de gerar a resposta, o sistema busca em uma base própria os "
                    "trechos mais relevantes para a pergunta e entrega esses trechos ao modelo "
                    "como contexto. A resposta passa a se apoiar em fontes verificáveis.",
                    "Um benefício prático é a atualização barata. Para incluir um artigo novo, "
                    "basta indexá-lo. Não é preciso treinar o modelo outra vez.",
                ],
            ),
            (
                "2. Etapas de um pipeline de RAG",
                [
                    "A ingestão lê os arquivos e extrai o texto de cada página. Páginas sem texto, "
                    "como imagens digitalizadas sem reconhecimento de caracteres, não geram conteúdo.",
                    "O chunking divide o texto em trechos de tamanho limitado, com uma pequena "
                    "sobreposição entre trechos vizinhos. A sobreposição evita que uma ideia seja "
                    "cortada exatamente no ponto em que a frase termina.",
                    "A indexação transforma cada trecho em um vetor numérico, chamado embedding, "
                    "e guarda os vetores em um índice. Textos com significado parecido ficam "
                    "próximos no espaço vetorial.",
                    "A recuperação transforma a pergunta em um vetor e procura os K trechos mais "
                    "similares, normalmente pela similaridade do cosseno. O valor de K controla o "
                    "equilíbrio entre cobertura e ruído.",
                    "A geração monta um prompt com os trechos encontrados e a pergunta, e pede "
                    "ao modelo que responda usando somente esse material, citando arquivo e página.",
                ],
            ),
            (
                "3. Limitações e cuidados",
                [
                    "Se a recuperação traz trechos errados, a resposta sai errada, mesmo com um "
                    "bom modelo. Por isso a qualidade do chunking e dos embeddings pesa tanto quanto "
                    "a qualidade do modelo de geração.",
                    "Perguntas que exigem comparar vários documentos precisam de uma estratégia "
                    "própria, como buscar separadamente em cada arquivo mencionado.",
                    "Respostas de sistemas de IA podem conter erros. Quem usa o resultado deve "
                    "conferir as citações nos documentos originais antes de aproveitá-lo.",
                    "Documentos sensíveis exigem atenção: ao usar um provedor externo, os trechos "
                    "saem da máquina do usuário. Para esses casos, modelos locais são mais seguros.",
                ],
            ),
        ],
    },
    {
        "arquivo": "testes_de_software.pdf",
        "titulo": "Fundamentos de Testes de Software",
        "secoes": [
            (
                "1. Por que testar",
                [
                    "Testes são a principal rede de segurança de um projeto. Eles documentam o "
                    "comportamento esperado do código e avisam quando uma mudança quebra algo "
                    "que funcionava.",
                    "Quanto mais cedo um defeito é encontrado, menor é o custo de corrigi-lo. Um erro "
                    "descoberto em produção costuma exigir investigação, correção urgente e "
                    "comunicação com usuários afetados.",
                ],
            ),
            (
                "2. Níveis de teste",
                [
                    "Testes unitários verificam uma função ou classe isolada. São rápidos e "
                    "apontam com precisão onde está o problema.",
                    "Testes de integração verificam a colaboração entre componentes, como o acesso "
                    "a um banco de dados ou a chamada de um serviço externo simulado.",
                    "Testes de ponta a ponta percorrem o sistema como um usuário faria. Dão muita "
                    "confiança, mas são mais lentos e mais frágeis. Por isso, em geral, existem "
                    "poucos deles em comparação com os unitários.",
                ],
            ),
            (
                "3. Testes baseados em propriedades",
                [
                    "Em vez de escolher exemplos à mão, o teste de propriedade declara uma regra "
                    "que deve valer para qualquer entrada válida, e uma biblioteca gera centenas "
                    "de casos aleatórios para tentar violá-la.",
                    "Quando a regra falha, a biblioteca reduz a entrada ao menor exemplo que ainda "
                    "reproduz o erro. Isso facilita muito o diagnóstico.",
                    "Boas propriedades descrevem relações, como a ida e volta entre gravar e ler, "
                    "ou a ordenação de um resultado. Elas complementam os testes de exemplo, que "
                    "cobrem casos de borda conhecidos.",
                ],
            ),
            (
                "4. Dublês de teste",
                [
                    "Dublês substituem dependências lentas ou não determinísticas, como serviços "
                    "de rede. Um dublê deve ser simples e previsível. Se ele ficar complexo demais, "
                    "o teste passa a verificar o dublê, e não o código real.",
                    "Vale usar dublês nas fronteiras do sistema e manter o núcleo de regras de "
                    "negócio testado com objetos reais.",
                ],
            ),
        ],
    },
    {
        "arquivo": "arquitetura_de_software.pdf",
        "titulo": "Arquitetura em Camadas e Qualidade de Software",
        "secoes": [
            (
                "1. Separação de responsabilidades",
                [
                    "Uma boa arquitetura divide o sistema em partes com responsabilidades claras. "
                    "Cada parte muda por um único motivo, o que reduz o efeito colateral de "
                    "uma alteração.",
                    "Na arquitetura em camadas, o núcleo com as regras de negócio não conhece "
                    "detalhes de banco de dados, de interface ou de serviços externos. As "
                    "dependências apontam sempre para o centro.",
                ],
            ),
            (
                "2. Interfaces e adaptadores",
                [
                    "O núcleo define interfaces para o que precisa do mundo externo, e as camadas "
                    "de fora fornecem as implementações. Trocar um provedor passa a ser uma "
                    "mudança local, sem reescrever as regras.",
                    "Um exemplo é um sistema que usa um provedor de embeddings. Com uma interface "
                    "única, é possível alternar entre um modelo local e um serviço externo "
                    "mudando apenas a configuração.",
                    "Essa mesma separação facilita os testes, pois os adaptadores reais podem ser "
                    "trocados por versões simples e determinísticas.",
                ],
            ),
            (
                "3. Atributos de qualidade",
                [
                    "Manutenibilidade indica a facilidade de entender e alterar o código. "
                    "Nomes claros, funções curtas e módulos coesos ajudam.",
                    "Desempenho e escalabilidade descrevem como o sistema se comporta com mais "
                    "dados ou mais usuários. Um índice vetorial em memória atende bem a algumas "
                    "dezenas de milhares de trechos. Acima disso, vale considerar uma solução "
                    "especializada.",
                    "Segurança inclui guardar segredos fora do código, validar entradas nas "
                    "fronteiras e evitar que credenciais apareçam em logs.",
                ],
            ),
            (
                "4. Decisões e trade-offs",
                [
                    "Toda decisão de arquitetura troca um benefício por um custo. Registrar o "
                    "motivo de cada escolha ajuda quem chegar depois a entender o contexto e a "
                    "revisar a decisão quando as condições mudarem.",
                    "Simplicidade costuma vencer. Começar com a solução mais simples que atende "
                    "aos requisitos e evoluir quando houver necessidade real evita complexidade "
                    "desnecessária.",
                ],
            ),
        ],
    },
]


def _estilos() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle("Titulo", parent=base["Title"], fontSize=20, leading=26),
        "secao": ParagraphStyle(
            "Secao", parent=base["Heading2"], fontSize=14, leading=18, spaceAfter=10
        ),
        "texto": ParagraphStyle(
            "Texto", parent=base["BodyText"], fontSize=11, leading=16, spaceAfter=8
        ),
        "nota": ParagraphStyle("Nota", parent=base["Italic"], fontSize=9, leading=12),
    }


def gerar_pdf(documento: dict, destino: Path) -> None:
    """Gera um PDF; cada seção começa em uma página nova, garantindo mais de uma página."""
    estilos = _estilos()
    pdf = SimpleDocTemplate(
        str(destino),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        title=documento["titulo"],
        author="Autor do projeto PDF RAG Chatbot",
    )
    elementos = [
        Paragraph(documento["titulo"], estilos["titulo"]),
        Paragraph(LICENCA, estilos["nota"]),
        Spacer(1, 0.8 * cm),
    ]
    for indice, (titulo, paragrafos) in enumerate(documento["secoes"]):
        if indice > 0:
            elementos.append(PageBreak())
        elementos.append(Paragraph(titulo, estilos["secao"]))
        elementos.extend(Paragraph(p, estilos["texto"]) for p in paragrafos)
    pdf.build(elementos)


def main() -> None:
    DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)
    for documento in DOCUMENTOS:
        destino = DIRETORIO_SAIDA / documento["arquivo"]
        gerar_pdf(documento, destino)
        print(f"Gerado: {destino} ({destino.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
