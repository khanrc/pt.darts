"""
Requirements:
    - GraphViz (apt-get + pip)
    - ImageMagick (apt-get)
    - PIL (pip)
"""
from graphviz import Digraph
import glob
import os
import subprocess
from PIL import Image
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("src_dir", help="dir of original dot files")
parser.add_argument("dst_dir", help="destination of dot files & png files")
parser.add_argument("gif_prefix", help="final gif prefix (path with file prefix)")
args = parser.parse_args()

if not os.path.exists(args.dst_dir):
    os.makedirs(args.dst_dir)


def glob_with_nonext(pattern):
    # glob + non-ext + sort
    dots = glob.glob(pattern)
    dots = list(filter(lambda p: not os.path.splitext(p)[1], dots))
    dots = sorted(dots)

    return dots


def add_epoch_caption(src_dir, dst_dir):
    """ add epoch to dots
    """
    print("Add epoch caption to dots ...")
    dots = glob_with_nonext(os.path.join(src_dir, "*"))

    out_paths = []
    for path in dots:
        print(path)
        fn = os.path.basename(path)
        ep = int(fn[2:4])
        out_path = os.path.join(dst_dir, fn)
        with open(path) as f:
            r = f.readlines()
            r.insert(-1,
                     "\tfontname=times fontsize=20 height=1.5 label=\"Epoch {:02d}\" "
                     "overlap=false\n".format(ep+1))
            text = "".join(r)
            with open(out_path, "w") as fout:
                fout.write(text)

            # convert dot to png
            png_path = out_path + ".png"
            subprocess.call("dot -Tpng {dot} -o {png}".format(dot=out_path, png=png_path), shell=True)

            out_paths.append(out_path)

    return out_paths


def to_square(dir_path):
    """ Re-sizing: adjust png size to square (max_size x max_size)

    Arguments:
        paths: dot file paths. (dot_path + '.png' == png_path)
    """
    dot_paths = glob_with_nonext(os.path.join(dir_path, "*"))

    # get max size
    max_w = 0
    max_h = 0
    for path in dot_paths:
        png_path = path + ".png"
        img = Image.open(png_path)
        w, h = img.size
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    # re-size
    w, h = max_w, max_h
    extent = "{}x{}".format(w, h)
    print("\nRe-size to {} ...".format(extent))
    for path in dot_paths:
        print(path)

        png_path = path + ".png"
        final_path = path + "-maxsize.png"
        subprocess.call("convert {png} -gravity center -background white "
                        "-extent {extent} {out}".format(
                            png=png_path, out=final_path, extent=extent),
                        shell=True)


def to_gif(dst_dir, gif_prefix):
    # Convert to GIF
    print("\nConvert to gif ...")
    st = time.time()
    print("Normal ... ", end="")
    cmd = "convert -resize 40% -delay 30 -loop 0 {target_glob} {output_path}".format(
        target_glob=os.path.join(dst_dir, "*-normal-maxsize.png"),
        output_path=gif_prefix+"-normal.gif")
    subprocess.call(cmd, shell=True)

    print("{:.0f}s".format(time.time() - st))
    st = time.time()
    print("Reduce ... ", end="")
    subprocess.call("convert -resize 40% -delay 30 -loop 0 {target_glob} {output_path}".format(
                        target_glob=os.path.join(dst_dir, "*-reduce-maxsize.png"),
                        output_path=gif_prefix+"-reduce.gif"),
                    shell=True)
    print("{:.0f}s".format(time.time() - st))
    print("DONE !")


if __name__ == "__main__":
    #add_epoch_caption(args.src_dir, args.dst_dir)
    to_square(args.dst_dir)
    to_gif(args.dst_dir, args.gif_prefix)
