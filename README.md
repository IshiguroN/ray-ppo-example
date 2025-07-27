# ray-ppo-example

このリポジトリは、VSCodeの[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)拡張機能に対応しています。  
リポジトリをフォークし、VSCodeで「Reopen in Container」を選択することで、必要な依存関係が自動でインストールされた開発環境が構築されます。


**本リポジトリは、UbuntuまたはWSL2環境上で、nvidia-dockerの利用を想定しています。**


## 環境構築手順

1. 本リポジトリをGitHub上でフォークします。
2. フォークしたリポジトリをローカルにクローンします。
3. VSCodeでクローンしたフォルダを開きます。
4. コマンドパレット（`Ctrl+Shift+P`）で「Dev Containers: Reopen in Container」を選択します。
5. 自動的にDockerコンテナ内で開発環境がセットアップされます。

詳細は[.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)および[Dockerfile](Dockerfile)をご参照

## BipedalWalker-v3 ([Gymnasium](https://gymnasium.farama.org/))
![demoGIF](./materials/video_BipedalWalker-v3.gif)