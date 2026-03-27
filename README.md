# Kojima-I2I-Translation-Network-using-ViT

## 研究概要
ARにおける画像変換ネットワークを用いた金属質感表現技術の研究

目的：部分的な背景画像＋仮想オブジェクトの法線画像から、
実際の照明を反映した金属質感を再現する。

## 使用技術
- Vision Transformer (ViT)：長距離依存性を捉え、照明情報を抽出
- PyTorch / Python
- 合成データセット作成＋定量評価

## 成果
- 従来手法を3つの定量評価指標で上回る結果を達成
- ARで環境情報が少ない場合でも、自然な金属表現を実現

## フォルダ構成

pix2pix_vggloss.py 提案手法の学習・評価を一括実行するメインスクリプト
models_vgg.py U-Net生成器、PatchGAN識別器、VGG特徴抽出ネットワーク
model_mynet.py ViT組み込みの提案Generatorネットワーク
datasets.py 画像ペア/HDF5読み込み、学習用データセット生成
utils.py 学習ログ・評価可視化・サンプル画像保存の補助関数

utils/ 出力画像比較・データ前処理・実験用補助スクリプト群
datasets/ 屋内外環境の合成データセット（indoor/outdoor）と作成スクリプト

## 作者
小島 魁斗 / Kaito Kojima
埼玉大学大学院 理工学研究科 情報工学プログラム M1
SIer業界志望（SE・インフラ・セキュリティ）
# Test
