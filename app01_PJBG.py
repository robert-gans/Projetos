import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import math
import os
from PIL import Image
import io
import re

# Configuração da página
st.set_page_config(
    page_title="Análise de Preço Justo de Graham - Gans Academy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para autenticação simples
def check_password():
    """Retorna `True` se a senha estiver correta."""
    def password_entered():
        """Verifica se a senha enviada pelo formulário está correta."""
        if st.session_state["password"] == "gansacademy":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Não armazene a senha
        else:
            st.session_state["password_correct"] = False

    # Retorna True se a senha estiver correta ou se já foi autenticado
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True

    # Mostra o formulário de entrada de senha
    st.image("logo.png", width=300)
    st.title("Análise de Preço Justo de Graham")
    st.markdown("### Acesso Restrito")
    st.text_input(
        "Digite a senha para acessar a aplicação", 
        type="password", 
        key="password",
        on_change=password_entered
    )
    
    if "password_correct" in st.session_state:
        if not st.session_state["password_correct"]:
            st.error("Senha incorreta. Tente novamente.")
    
    return False

# Função para calcular o preço justo de Graham
def calcular_preco_justo(lpa, vpa):
    # Verificar se LPA ou VPA são negativos ou zero
    if pd.isna(lpa) or pd.isna(vpa) or lpa <= 0 or vpa <= 0:
        return 0
    return math.sqrt(22.5 * lpa * vpa)

# Função para encontrar colunas por nome aproximado
def encontrar_coluna(df, nome_coluna):
    # Primeiro, verifica se a coluna existe exatamente como especificada
    if nome_coluna in df.columns:
        return nome_coluna
    
    # Verifica se existe uma coluna com espaços extras
    espacos_possiveis = [f" {nome_coluna}", f"{nome_coluna} ", f" {nome_coluna} "]
    for col in espacos_possiveis:
        if col in df.columns:
            return col
    
    # Verifica se existe uma coluna que contém o nome (case insensitive)
    for col in df.columns:
        if nome_coluna.lower() in col.lower():
            return col
    
    # Se não encontrar, retorna None
    return None

# Função para limpar nomes de colunas
def limpar_nomes_colunas(df):
    df.columns = [col.strip() for col in df.columns]
    return df

# Função para converter string numérica em formato brasileiro para float
def converter_numero_br(valor):
    if pd.isna(valor) or valor == '':
        return 0.0
    
    # Se já for um número, retorna ele mesmo
    if isinstance(valor, (int, float)):
        return float(valor)
    
    # Converte para string caso não seja
    valor_str = str(valor)
    
    # Remove caracteres não numéricos, exceto ponto e vírgula
    valor_str = re.sub(r'[^\d.,\-]', '', valor_str)
    
    # Verifica se é um formato com ponto como separador de milhar e vírgula como decimal
    # Exemplo: 1.234,56
    if ',' in valor_str and '.' in valor_str and valor_str.rindex('.') < valor_str.rindex(','):
        # Remove pontos e substitui vírgula por ponto
        valor_str = valor_str.replace('.', '').replace(',', '.')
    
    # Verifica se é um formato com ponto como separador de milhar e também como decimal
    # Exemplo: 2.930.92
    elif '.' in valor_str and valor_str.count('.') > 1:
        # Mantém apenas o último ponto como decimal
        ultimo_ponto = valor_str.rindex('.')
        valor_str = valor_str[:ultimo_ponto].replace('.', '') + valor_str[ultimo_ponto:]
    
    # Verifica se é um formato com vírgula como decimal
    # Exemplo: 1234,56
    elif ',' in valor_str:
        # Substitui vírgula por ponto
        valor_str = valor_str.replace(',', '.')
    
    try:
        return float(valor_str)
    except ValueError:
        st.error(f"Não foi possível converter o valor '{valor}' para número. Formato não reconhecido.")
        return 0.0

# Função para processar os dados da planilha
def processar_dados(df):
    # Exibir informações de debug
    st.write("Colunas encontradas na planilha:")
    st.write(df.columns.tolist())
    
    # Limpar nomes de colunas
    df = limpar_nomes_colunas(df)
    
    # Exibir informações de debug após limpeza
    st.write("Colunas após limpeza:")
    st.write(df.columns.tolist())
    
    # Encontrar colunas necessárias
    col_ticker = encontrar_coluna(df, "TICKER")
    col_preco = encontrar_coluna(df, "PRECO")
    col_vpa = encontrar_coluna(df, "VPA")
    col_lpa = encontrar_coluna(df, "LPA")
    col_valor_mercado = encontrar_coluna(df, "VALOR DE MERCADO")
    
    # Verificar se todas as colunas foram encontradas
    colunas_faltantes = []
    if not col_ticker:
        colunas_faltantes.append("TICKER")
    if not col_preco:
        colunas_faltantes.append("PRECO")
    if not col_vpa:
        colunas_faltantes.append("VPA")
    if not col_lpa:
        colunas_faltantes.append("LPA")
    
    if colunas_faltantes:
        st.error(f"As seguintes colunas não foram encontradas na planilha: {', '.join(colunas_faltantes)}")
        return None
    
    # Converter colunas para o formato correto
    df[col_ticker] = df[col_ticker].astype(str)
    
    # Mostrar exemplos de valores para debug
    st.write("Exemplos de valores nas colunas numéricas:")
    for col in [col_preco, col_vpa, col_lpa]:
        st.write(f"Coluna {col}: {df[col].head().tolist()}")
    
    # Converter colunas numéricas usando a função personalizada
    try:
        df[col_preco] = df[col_preco].apply(converter_numero_br)
        df[col_vpa] = df[col_vpa].apply(converter_numero_br)
        df[col_lpa] = df[col_lpa].apply(converter_numero_br)
        if col_valor_mercado:
            df[col_valor_mercado] = df[col_valor_mercado].apply(converter_numero_br)
    except Exception as e:
        st.error(f"Erro ao converter valores numéricos: {str(e)}")
        st.write("Verifique se os valores estão em um formato numérico válido.")
        return None
    
    # Registrar o número total de ações antes da limpeza
    total_acoes_inicial = len(df)
    st.write(f"Total de ações na planilha original: {total_acoes_inicial}")
    
    # Criar cópias das colunas originais para referência
    df['PRECO_ORIGINAL'] = df[col_preco]
    df['VPA_ORIGINAL'] = df[col_vpa]
    df['LPA_ORIGINAL'] = df[col_lpa]
    
    # Identificar registros com valores inválidos
    acoes_preco_invalido = df[df[col_preco] <= 0].shape[0]
    acoes_vpa_invalido = df[df[col_vpa] <= 0].shape[0]
    acoes_lpa_invalido = df[df[col_lpa] <= 0].shape[0]
    
    # Filtrar registros com valores válidos para análise de Graham
    df_valido = df[(df[col_preco] > 0) & (df[col_vpa] > 0) & (df[col_lpa] > 0)].copy()
    
    # Registrar o número de ações após a limpeza
    total_acoes_validas = len(df_valido)
    total_acoes_removidas = total_acoes_inicial - total_acoes_validas
    
    # Exibir resumo da limpeza
    st.write("### Resumo da Limpeza de Dados")
    st.write(f"Total de ações removidas: {total_acoes_removidas} ({total_acoes_removidas/total_acoes_inicial*100:.1f}%)")
    st.write(f"- Ações com preço zero ou negativo: {acoes_preco_invalido}")
    st.write(f"- Ações com VPA zero ou negativo: {acoes_vpa_invalido}")
    st.write(f"- Ações com LPA zero ou negativo: {acoes_lpa_invalido}")
    st.write(f"Total de ações válidas para análise: {total_acoes_validas} ({total_acoes_validas/total_acoes_inicial*100:.1f}%)")
    
    # Calcular o preço justo de Graham apenas para ações válidas
    df_valido['PRECO_JUSTO_GRAHAM'] = df_valido.apply(lambda row: calcular_preco_justo(row[col_lpa], row[col_vpa]), axis=1)
    
    # Determinar a avaliação
    df_valido['AVALIACAO'] = df_valido.apply(
        lambda row: 'Subavaliada' if row[col_preco] <= row['PRECO_JUSTO_GRAHAM'] else 'Sobreavaliada', 
        axis=1
    )
    
    # Calcular a diferença percentual
    df_valido['DIFERENCA_PERCENTUAL'] = df_valido.apply(
        lambda row: ((row[col_preco] - row['PRECO_JUSTO_GRAHAM']) / row['PRECO_JUSTO_GRAHAM'] * 100), 
        axis=1
    )
    
    # Adicionar valor de mercado se disponível
    if col_valor_mercado:
        df_valido['VALOR_MERCADO'] = df_valido[col_valor_mercado]
    else:
        # Se não tiver valor de mercado, usar um valor padrão
        df_valido['VALOR_MERCADO'] = 1000000000  # 1 bilhão como padrão
    
    # Renomear colunas para melhor visualização
    colunas_resultado = [col_ticker, col_preco, col_lpa, col_vpa, 'PRECO_JUSTO_GRAHAM', 'AVALIACAO', 'DIFERENCA_PERCENTUAL']
    if col_valor_mercado:
        colunas_resultado.append('VALOR_MERCADO')
    
    df_resultado = df_valido[colunas_resultado].copy()
    
    nomes_colunas = [
        'Ticker', 
        'Preço Atual', 
        'Lucro por Ação (LPA)', 
        'Valor Patrimonial por Ação (VPA)', 
        'Preço Justo Graham', 
        'Avaliação', 
        'Diferença (%)'
    ]
    
    if col_valor_mercado:
        nomes_colunas.append('Valor de Mercado')
    
    df_resultado.columns = nomes_colunas
    
    return df_resultado

# Função para criar gráfico de comparação
def criar_grafico_comparacao(df, tickers_selecionados=None):
    if tickers_selecionados:
        df_plot = df[df['Ticker'].isin(tickers_selecionados)].copy()
    else:
        df_plot = df.copy()
    
    # Ordenar por diferença percentual
    df_sorted = df_plot.sort_values(by='Diferença (%)')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Criar barras para preço atual e preço justo
    x = np.arange(len(df_sorted))
    width = 0.35
    
    # Cores baseadas na avaliação
    colors_atual = ['#FF6B6B' if aval == 'Sobreavaliada' else '#4CAF50' for aval in df_sorted['Avaliação']]
    colors_justo = ['#FFA07A' if aval == 'Sobreavaliada' else '#8BC34A' for aval in df_sorted['Avaliação']]
    
    ax.bar(x - width/2, df_sorted['Preço Atual'], width, label='Preço Atual', color=colors_atual)
    ax.bar(x + width/2, df_sorted['Preço Justo Graham'], width, label='Preço Justo', color=colors_justo)
    
    # Adicionar rótulos e título
    ax.set_xlabel('Ticker', fontweight='bold', fontsize=12)
    ax.set_ylabel('Preço (R$)', fontweight='bold', fontsize=12)
    ax.set_title('Comparação entre Preço Atual e Preço Justo de Graham', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['Ticker'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    
    # Adicionar valores percentuais acima das barras
    for i, ticker in enumerate(df_sorted['Ticker']):
        diferenca = df_sorted['Diferença (%)'].iloc[i]
        ax.annotate(f'{diferenca:.1f}%', 
                   xy=(i, max(df_sorted['Preço Atual'].iloc[i], df_sorted['Preço Justo Graham'].iloc[i]) + 1),
                   ha='center', va='bottom',
                   fontweight='bold',
                   color='red' if diferenca > 0 else 'green')
    
    plt.tight_layout()
    
    return fig

# Função para criar gráfico de pizza para distribuição de avaliações
def criar_grafico_pizza(df):
    # Contar número de ações subavaliadas e sobreavaliadas
    contagem = df['Avaliação'].value_counts()
    
    # Criar gráfico de pizza
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#4CAF50', '#FF6B6B']  # Verde para subavaliadas, vermelho para sobreavaliadas
    
    wedges, texts, autotexts = ax.pie(
        contagem, 
        labels=contagem.index, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=[0.05, 0] if 'Subavaliada' in contagem.index else [0, 0.05],
        shadow=True
    )
    
    # Estilizar textos
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_title('Distribuição de Ações por Avaliação', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    return fig

# Função para criar gráfico de dispersão
def criar_grafico_dispersao(df):
    # Verificar se temos a coluna de valor de mercado
    if 'Valor de Mercado' not in df.columns:
        # Usar um valor padrão se não tivermos o valor de mercado
        df['Valor de Mercado'] = 1000000000  # 1 bilhão como padrão
    
    # Normalizar o tamanho dos pontos (valor de mercado)
    valor_mercado_max = df['Valor de Mercado'].max()
    df['Tamanho'] = df['Valor de Mercado'] / valor_mercado_max * 100
    
    # Garantir um tamanho mínimo para visualização
    df['Tamanho'] = df['Tamanho'].apply(lambda x: max(x, 5))
    
    # Criar o gráfico de dispersão com Plotly
    fig = px.scatter(
        df,
        x='Ticker',
        y='Diferença (%)',
        size='Tamanho',
        color='Avaliação',
        hover_name='Ticker',
        color_discrete_map={'Subavaliada': '#4CAF50', 'Sobreavaliada': '#FF6B6B'},
        hover_data={
            'Ticker': True,
            'Preço Atual': True,
            'Preço Justo Graham': True,
            'Diferença (%)': ':.2f',
            'Valor de Mercado': ':,.2f',
            'Tamanho': False  # Não mostrar a coluna de tamanho normalizado
        },
        title='Dispersão de Ações por Diferença Percentual',
        height=600
    )
    
    # Adicionar linha horizontal em 0%
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        xref='paper',
        line=dict(color='black', width=2, dash='dash')
    )
    
    # Personalizar o layout
    fig.update_layout(
        xaxis_title='Ticker',
        yaxis_title='Diferença (%)',
        legend_title='Avaliação',
        font=dict(size=12),
        xaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Adicionar anotações explicativas
    fig.add_annotation(
        x=0.5,
        y=50,
        xref='paper',
        text="Sobreavaliadas",
        showarrow=False,
        font=dict(size=14, color="#FF6B6B")
    )
    
    fig.add_annotation(
        x=0.5,
        y=-50,
        xref='paper',
        text="Subavaliadas",
        showarrow=False,
        font=dict(size=14, color="#4CAF50")
    )
    
    return fig

# Função para converter o DataFrame para CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para aplicar estilo à tabela
def highlight_rows(s):
    if s['Avaliação'] == 'Sobreavaliada':
        return ['background-color: #FFCCCC'] * len(s)
    else:
        return ['background-color: #CCFFCC'] * len(s)

# Aplicação principal
def main():
    # Carregar logo
    logo = Image.open("logo.png")
    
    # Sidebar com logo
    st.sidebar.image(logo, width=200)
    st.sidebar.title("Gans Academy")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sobre a Análise")
    st.sidebar.markdown("""
    Esta aplicação utiliza a fórmula de Benjamin Graham para calcular o preço justo de ações.
    
    **Fórmula utilizada:**
    ```
    Preço Justo = √(22,5 × LPA × VPA)
    ```
    
    Onde:
    - LPA: Lucro por Ação
    - VPA: Valor Patrimonial por Ação
    
    **Importante:** A fórmula só é aplicável para ações com LPA e VPA positivos.
    Ações com valores negativos ou zero são excluídas da análise.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Como usar")
    st.sidebar.markdown("""
    1. Acesse o site [Status Invest](https://statusinvest.com.br/)
    2. Vá para a seção de "Busca Avançada" de ações
    3. Configure os filtros desejados
    4. Clique em "Buscar"
    5. No resultado, clique em "Exportar" para baixar a planilha CSV
    6. Faça upload da planilha nesta aplicação
    7. Analise os resultados e filtre conforme necessário
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Desenvolvido por Gans Academy")
    st.sidebar.markdown("[gans-academy.com](https://gans-academy.com)")
    
    # Título principal
    st.title("Análise de Preço Justo de Graham")
    st.markdown("### Descubra se suas ações estão sub ou sobreavaliadas")
    
    # Opções de upload
    st.subheader("Upload da planilha")
    st.markdown("""
    Faça upload da planilha CSV exportada do Status Invest. 
    A aplicação foi otimizada para lidar com diferentes formatos de planilha e números.
    
    **Nota importante:** Ações com LPA ou VPA negativos ou zero serão excluídas da análise,
    pois a fórmula de Graham não é aplicável nesses casos.
    """)
    
    # Upload da planilha
    uploaded_file = st.file_uploader("Faça upload da planilha CSV do Status Invest", type=["csv"])
    
    # Opções de separador
    separador = st.selectbox(
        "Selecione o separador usado na planilha:",
        options=[";", ",", "\\t", "|"],
        index=0,
        help="O separador padrão do Status Invest é ponto e vírgula (;)"
    )
    
    # Mapeamento de separadores
    separadores = {
        ";": ";",
        ",": ",",
        "\\t": "\t",
        "|": "|"
    }
    
    if uploaded_file is not None:
        try:
            # Exibir informações sobre o arquivo
            st.write(f"Arquivo carregado: {uploaded_file.name}")
            
            # Ler as primeiras linhas para debug
            conteudo = uploaded_file.read().decode('utf-8')
            primeiras_linhas = "\n".join(conteudo.split("\n")[:5])
            st.write("Primeiras linhas do arquivo:")
            st.code(primeiras_linhas)
            
            # Resetar o ponteiro do arquivo
            uploaded_file.seek(0)
            
            # Ler o arquivo CSV com o separador selecionado
            df = pd.read_csv(uploaded_file, sep=separadores[separador])
            
            # Processar os dados
            df_resultados = processar_dados(df)
            
            if df_resultados is not None and not df_resultados.empty:
                # Exibir informações sobre os dados
                st.success(f"Análise concluída para {len(df_resultados)} ações válidas!")
                
                # Opções de filtragem
                st.subheader("Filtros")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    filtro_avaliacao = st.multiselect(
                        "Filtrar por avaliação:",
                        options=["Todas", "Subavaliada", "Sobreavaliada"],
                        default=["Todas"]
                    )
                
                with col2:
                    min_preco = st.number_input("Preço mínimo:", min_value=0.0, value=0.0, step=1.0)
                    max_preco = st.number_input("Preço máximo:", min_value=0.0, value=1000.0, step=10.0)
                
                with col3:
                    min_diferenca = st.number_input("Diferença % mínima:", min_value=-100.0, value=-100.0, step=5.0)
                    max_diferenca = st.number_input("Diferença % máxima:", min_value=-100.0, value=100.0, step=5.0)
                
                # Aplicar filtros
                df_filtrado = df_resultados.copy()
                
                if "Todas" not in filtro_avaliacao:
                    df_filtrado = df_filtrado[df_filtrado["Avaliação"].isin(filtro_avaliacao)]
                
                df_filtrado = df_filtrado[
                    (df_filtrado["Preço Atual"] >= min_preco) & 
                    (df_filtrado["Preço Atual"] <= max_preco) &
                    (df_filtrado["Diferença (%)"] >= min_diferenca) &
                    (df_filtrado["Diferença (%)"] <= max_diferenca)
                ]
                
                # Exibir resultados
                st.subheader("Resultados")
                
                if df_filtrado.empty:
                    st.warning("Nenhuma ação encontrada com os filtros selecionados.")
                else:
                    # Formatar a tabela
                    df_display = df_filtrado.copy()
                    
                    # Formatar colunas numéricas
                    df_display['Preço Atual'] = df_display['Preço Atual'].map('R$ {:.2f}'.format)
                    df_display['Lucro por Ação (LPA)'] = df_display['Lucro por Ação (LPA)'].map('R$ {:.2f}'.format)
                    df_display['Valor Patrimonial por Ação (VPA)'] = df_display['Valor Patrimonial por Ação (VPA)'].map('R$ {:.2f}'.format)
                    df_display['Preço Justo Graham'] = df_display['Preço Justo Graham'].map('R$ {:.2f}'.format)
                    df_display['Diferença (%)'] = df_display['Diferença (%)'].map('{:.2f}%'.format)
                    
                    if 'Valor de Mercado' in df_display.columns:
                        df_display['Valor de Mercado'] = df_display['Valor de Mercado'].map('R$ {:.2f}'.format)
                    
                    # Aplicar estilo à tabela
                    st.dataframe(df_display.style.apply(highlight_rows, axis=1))
                    
                    # Visualizações gráficas
                    st.subheader("Visualizações Gráficas")
                    
                    # Gráfico de dispersão
                    st.write("### Gráfico de Dispersão (Diferença Percentual)")
                    st.write("""
                    Este gráfico mostra a distribuição das ações de acordo com a diferença percentual entre o preço atual e o preço justo.
                    O tamanho de cada ponto é proporcional ao valor de mercado da empresa.
                    Pontos acima da linha central (0%) representam ações sobreavaliadas, enquanto pontos abaixo representam ações subavaliadas.
                    """)
                    
                    fig_dispersao = criar_grafico_dispersao(df_filtrado)
                    st.plotly_chart(fig_dispersao, use_container_width=True)
                    
                    # Seleção de tickers para visualização gráfica de barras
                    st.write("### Gráfico de Comparação (Preço Atual vs Preço Justo)")
                    
                    # Limitar a 20 tickers para melhor visualização
                    if len(df_filtrado) > 20:
                        st.info("Selecione até 20 tickers para visualização gráfica:")
                        tickers_selecionados = st.multiselect(
                            "Tickers:",
                            options=df_filtrado["Ticker"].tolist(),
                            default=df_filtrado.sort_values(by="Diferença (%)")["Ticker"].head(10).tolist()
                        )
                    else:
                        tickers_selecionados = df_filtrado["Ticker"].tolist()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if tickers_selecionados:
                            fig_barras = criar_grafico_comparacao(df_filtrado, tickers_selecionados)
                        else:
                            fig_barras = criar_grafico_comparacao(df_filtrado.head(20))
                        st.pyplot(fig_barras)
                    
                    with col2:
                        fig_pizza = criar_grafico_pizza(df_filtrado)
                        st.pyplot(fig_pizza)
                    
                    # Botão para download
                    csv = convert_df_to_csv(df_filtrado)
                    st.download_button(
                        label="Baixar resultados como CSV",
                        data=csv,
                        file_name='analise_graham_filtrada.csv',
                        mime='text/csv',
                    )
                    
                    # Resumo da análise
                    st.subheader("Resumo da Análise")
                    total_acoes = len(df_filtrado)
                    acoes_subavaliadas = len(df_filtrado[df_filtrado['Avaliação'] == 'Subavaliada'])
                    acoes_sobreavaliadas = len(df_filtrado[df_filtrado['Avaliação'] == 'Sobreavaliada'])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total de Ações Analisadas", total_acoes)
                    col2.metric("Ações Subavaliadas", acoes_subavaliadas, f"{acoes_subavaliadas/total_acoes*100:.1f}%")
                    col3.metric("Ações Sobreavaliadas", acoes_sobreavaliadas, f"{acoes_sobreavaliadas/total_acoes*100:.1f}%")
                    
                    # Estatísticas adicionais
                    st.subheader("Estatísticas")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Ações mais subavaliadas")
                        df_sub = df_filtrado[df_filtrado['Avaliação'] == 'Subavaliada'].sort_values(by='Diferença (%)')
                        if not df_sub.empty:
                            st.table(df_sub.head(5)[['Ticker', 'Preço Atual', 'Preço Justo Graham', 'Diferença (%)']])
                        else:
                            st.info("Nenhuma ação subavaliada encontrada.")
                    
                    with col2:
                        st.markdown("#### Ações mais sobreavaliadas")
                        df_sobre = df_filtrado[df_filtrado['Avaliação'] == 'Sobreavaliada'].sort_values(by='Diferença (%)', ascending=False)
                        if not df_sobre.empty:
                            st.table(df_sobre.head(5)[['Ticker', 'Preço Atual', 'Preço Justo Graham', 'Diferença (%)']])
                        else:
                            st.info("Nenhuma ação sobreavaliada encontrada.")
        
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar a planilha: {str(e)}")
            st.markdown("""
            **Possíveis razões:**
            - Formato da planilha diferente do esperado
            - Problemas com os dados (valores ausentes, formato incorreto)
            - Erro interno da aplicação
            
            **Sugestões:**
            - Tente selecionar um separador diferente
            - Verifique se a planilha está no formato correto (CSV do Status Invest)
            - Certifique-se de que as colunas necessárias existem e estão preenchidas
            """)
    else:
        st.info("""
        ### Como obter a planilha do Status Invest:
        
        1. Acesse o site [Status Invest](https://statusinvest.com.br/)
        2. Vá para a seção de "Busca Avançada" de ações
        3. Configure os filtros desejados
        4. Clique em "Buscar"
        5. No resultado, clique em "Exportar" para baixar a planilha CSV
        6. Faça upload da planilha nesta aplicação
        
        A aplicação calculará automaticamente o preço justo de Graham para todas as ações na planilha.
        """)
        
        # Mostrar exemplo de análise
        # st.subheader("Exemplo de análise")
        # st.image("exemplo_analise.png", caption="Exemplo de análise com dados do Status Invest")

# Executar a aplicação
if __name__ == "__main__":
    if check_password():
        main()
