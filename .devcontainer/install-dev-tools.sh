# Clean up install files
pip3 cache purge 2>&1
sudo apt-get autoremove 2>&1
sudo apt-get clean 2>&1
