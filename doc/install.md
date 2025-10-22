
#### Mac Apple Sillicon

```
/bin/bash -c "$(curl -fsSL \
https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

brew install bazel

bazel --version

```

cd "/opt/homebrew/Cellar/bazel/8.4.2/libexec/bin" && curl -fLO https://releases.bazel.build/5.0.0/release/bazel-5.0.0-darwin-arm64 && chmod +x bazel-5.0.0-darwin-arm64 && cd -

### ubuntu x86_64

```
# 删除之前的编译
sudo rm -rf /tmp/Python-3.10.12
sudo rm -rf /usr/local/python3.10

cd /tmp
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
tar xzf Python-3.10.12.tgz
cd Python-3.10.12

# Configure WITHOUT --enable-optimizations to avoid compilation errors
./configure --enable-shared --prefix=/usr/local/python3.10 --with-ensurepip=install
make -j$(nproc)
sudo make install

# Update library path
echo "/usr/local/python3.10/lib" | sudo tee /etc/ld.so.conf.d/python3.10.conf
sudo ldconfig

# Verify
/usr/local/python3.10/bin/python3.10 --version
/usr/local/python3.10/bin/python3.10 -c "import sysconfig; print(sysconfig.get_path('include'))"


```