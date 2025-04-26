# 📈 Análise de Preço Justo de Graham

Bem-vindo à aplicação **Análise de Preço Justo de Graham** desenvolvida pela **Gans Academy**!

Esta aplicação web foi criada com **Python** e **Streamlit** para permitir que você analise ações de forma simples e eficiente, utilizando a famosa fórmula de Benjamin Graham para calcular o preço justo dos ativos.

## 🚀 Funcionalidades

- Upload de planilhas CSV do site **Status Invest**.
- Limpeza e tratamento automático dos dados (formatos brasileiros, valores inválidos).
- Cálculo do **Preço Justo de Graham**:
  \[
  Preço\,Justo = \sqrt{22,5 \times LPA \times VPA}
  \]
- Classificação automática das ações como **Subavaliadas** ou **Sobreavaliadas**.
- Visualizações gráficas:
  - Gráfico de dispersão (Plotly)
  - Gráfico de comparação entre preço atual e preço justo
  - Gráfico de pizza com a distribuição das avaliações
- Filtros dinâmicos de avaliação, preço e diferença percentual.
- Download do resultado final em CSV.
- Interface segura com autenticação por senha.

## 📥 Como usar

1. Baixe sua planilha de ações no [Status Invest](https://statusinvest.com.br/), na seção de "Busca Avançada".
2. Faça upload da planilha na aplicação.
3. Analise os resultados, visualize gráficos e baixe a análise final.

**Senha padrão para acesso:** `gansacademy`

## 🔧 Instalação local

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/seuusuario/Projetos.git
cd Projetos/preco-justo-graham
pip install -r requirements.txt
streamlit run app01_PJBG.py
