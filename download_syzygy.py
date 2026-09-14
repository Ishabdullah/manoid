import urllib.request
import re
import os

base_url = "http://tablebase.sesse.net/syzygy/3-4-5/"
print("Fetching directory listing...")
try:
    html = urllib.request.urlopen(base_url).read().decode('utf-8')
    files = re.findall(r'href="([^"]+\.rtb[wz])"', html)

    os.makedirs("syzygy_345", exist_ok=True)

    def piece_count(f):
        name = f.split('.')[0]
        return len(name.replace('v', ''))

    to_download = [f for f in files if piece_count(f) <= 3]
    print(f"Found {len(to_download)} files for <= 3 pieces.")

    for f in to_download:
        out_path = os.path.join("syzygy_345", f)
        if not os.path.exists(out_path):
            print(f"Downloading {f}...")
            urllib.request.urlretrieve(base_url + f, out_path)
    print("Done with 3 piece!")
except Exception as e:
    print("Failed:", e)
