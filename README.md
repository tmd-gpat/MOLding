## Installation

```sh
# make plugins
$ export PLUGINDIR=`pwd`/vmd-1.9.2/plugins/
$ cd plugins
$ make MACOSXX86_64 # change MACOSXX86_64 to your architecture
$ make distrib

# make vmd
$ cd ../vmd-1.9.2
$ vi configure # set $install_bin_dir and $install_library_dir properly
$ make macosx.x86_64.opengl.nocuda.notachyon # change macosx.x86_64.opengl.nocuda.notachyon to your favorite build option. To see the list of options, run "make"
$ cd src
$ make
$ make install

# run vmd
$ cd (install_bin_dir)
$ ./vmd
```

## Dependencies

The following libraries are required to build VMD program.

* tcl/tk
* fltk
