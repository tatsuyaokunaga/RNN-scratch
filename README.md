# RNN-scratch

## RNNとは

RNNとは、音声や映像、言語といった「時系列の流れに意味を持つデータ」の予測や分類に用いられる再帰型ネットワークのこと。  
再帰型ネットワークがあるタイムステップt-1で行った決定は、その次のタイムステップtで行う決定に影響する。  
従って、再帰型ネットワークとは、現在と近い過去の2か所から情報を受け取り、それらを組み合わせて新しい情報をどう処理するかを決めるネットワークである。  
言い方を変えると、RNNは以前に計算された情報を覚えるための記憶力を持っている。  
この自己回帰構造のおかげで前の情報を取り入れた解析が可能となり、音声認識等の時系列データ解析でその有効性が知られている。  
理論的にはRNNはとても長い文章の情報を利用することが可能だが、実際に実装してみると2,3ステップくらい前の情報しか覚えらない。  