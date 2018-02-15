#!/usr/bin/env bash

url_list="fall11_urls"
output_dir="/mnt/pccfs/not_backed_up/data/imagenet"
# download_count=100000
fraction_to_download=.01

echo "Downloading URL list..."
[[ -e /tmp/$url_list.txt ]] || curl --progress-bar "http://image-net.org/imagenet_data/urls/imagenet_$url_list.tgz" | tar xvz -C /tmp/

echo -e "\nDownloading images, this could take a while..."
awk -F '\t' "BEGIN {srand()} /jpg/ { if (rand() <= $fraction_to_download) printf \"%s%c%s%c\", \$1, 0, \$2, 0}" /tmp/$url_list.txt \
| xargs -0 -L 2 -P 12 \
bash -c "curl -fLs --create-dirs -o $output_dir/\$0 -w '%{filename_effective}\n' -C - \$1 || true" \
&& rm /tmp/$url_list
