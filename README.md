# DL-SAFE

DL-SAFE (Deep Learning-based SAFeguard for Edge botnet detection) é uma ferramenta para detecção de botnets em ambientes IoT usando aprendizado profundo.

Desenvolvido pela equipe GTA/UFRJ.

# Descrição

A ferramenta consiste em 4 módulos: o Módulo de Treinamento de Modelos, o Módulo de Captura de Dados, o Módulo de Tratamento de Dados e o Módulo de Processamento de Dados. Os componentes estão apresentados na Figura abaixo:

![Alt text](arquiteturaDLSAFE.PNG?raw=true "Arquitetura do DL-SAFE")

**Módulo de Treinamento de Modelos** - Cria os modelos de classificação utilizados pelo módulo de processamento de dados. O processo de treinamento inicia limpando o conjunto de dados BoT-IoT com a remoção de dados incompletos ou nulos. Os dados são então divididos em dois conjuntos de dados: um conjunto de treinamento, contendo 70% dos dados, e um conjunto de teste contendo os 30% restantes. O conjunto de treinamento é usado para obter os modelos de classificação; após a obtenção dos modelos, o desempenho de classificação é avaliado por meio do módulo de avaliação offline, utilizando o conjunto de testes. Obtém-se assim resultados de acurácia, precisão, sensibilidade, F1-Score e perda. Diferentemente dos outros módulos, o módulo de treinamento de modelos não precisa estar ativo durante a classificação de tráfego em tempo real; assim, é possível treinar os modelos com antecedência antes de utilizar a ferramenta para classificar dados em tempo real.

**Módulo de Captura de Dados** - Implementado pelo Open Argus. Este módulo lê o tráfego de rede e converte os dados em um arquivo ARGUS. Utilizando métodos do próprio Open Argus, um documento CSV é extraído do arquivo ARGUS contendo os fluxos e suas respectivas características em um formato legível pelo Módulo de Tratamento de Dados. O Módulo de Captura de Dados agrupa os dados de rede usando uma janela de tempo configurável, gerando por padrão um arquivo CSV a cada cinco segundos.

**Módulo de Tratamento de Dados** - Lê o arquivo CSV gerado pelo Módulo de Captura de Dados, extrai características adicionais para que o conjunto final possua as mesmas características utilizadas durante o treinamento, e realiza a limpeza de dados utilizando o mesmo método implementado no Módulo de Treinamento de Modelos. Este módulo aguarda até receber um arquivo CSV, realiza o tratamento desses dados e envia o arquivo CSV resultante para o módulo de processamento de dados.

**Módulo de Processamento de Dados** - Recebe os dados tratados no formato CSV, executa o modelo de classificação selecionado usando a biblioteca PyTorch e salva os resultados da classificação para análise posterior do usuário. O usuário pode empregar o modelo de classificação em seu formato regular ou, alternativamente, usar o formato de quantização pós-treinamento (PTQ) de 8 bits para obter melhor vazão em troca de um desempenho de classificação ligeiramente inferior.

A [wiki](https://github.com/GTA-UFRJ/neuralnetwork-IoT/wiki) da ferramenta inclui um guia detalhando o processo de instalação, assim como um manual sobre como configurar e executar o DL-SAFE.