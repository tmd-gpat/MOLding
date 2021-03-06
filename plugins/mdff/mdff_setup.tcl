############################################################################
#cr
#cr            (C) Copyright 1995-2009 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: mdff_setup.tcl,v $
#       $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
#       $Revision: 1.9 $       $Date: 2014/10/23 20:14:32 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>
#
# mdff gridpdb -- creates a pdb file docking
# mdff setup   -- writes a NAMD config file for docking
#

package require readcharmmpar
package require exectool
package provide mdff_setup 0.3

namespace eval ::MDFF::Setup:: {

  variable defaultDiel 80
  variable defaultScaling1_4 1.0
  variable defaultGScale 0.3
  variable defaultTemp 300
  variable defaultNumSteps 500000
  variable defaultMinimize 200
  variable defaultConsCol B
  variable defaultFixCol O
  variable defaultMargin 0
  variable defaultDir [pwd]
  variable defaultLite 0

  variable defaultK 10.0
  variable defaultGridpdbSel {noh and (protein or nucleic)}
  #xMDFF related variables
  variable defaultxMDFF    0
  variable defaultBFS      0
  variable defaultMask     0
  variable defaultRefSteps 20000
  variable defaultCrystPDB 0
  variable defaultMaskRes 5
  variable defaultMaskCutoff 5
  #IMD related variables
  variable defaultIMD 0
  variable defaultIMDPort 2000
  variable defaultIMDFreq 1
  variable defaultIMDWait "no"
  variable defaultIMDIgnore "no"
}

proc ::MDFF::Setup::mdff_setup_usage { } {
    
  variable defaultDiel
  variable defaultScaling1_4
  variable defaultGScale
  variable defaultTemp
  variable defaultParFile
  variable defaultNumSteps
  variable defaultMinimize
  variable defaultConsCol 
  variable defaultFixCol
  variable defaultMargin 
  variable defaultDir
  variable defaultLite

  #xMDFF related variables
  variable defaultxMDFF   
  variable defaultBFS      
  variable defaultMask     
  variable defaultRefSteps
  variable defaultCrystPDB 
  variable defaultMaskRes
  variable defaultMaskCutoff
  
  #IMD related variables 
  variable defaultIMD
  variable defaultIMDPort
  variable defaultIMDFreq
  variable defaultIMDWait
  variable defaultIMDIgnore
  ::MDFF::Setup::init_files

  puts "Usage: mdff setup -o <output prefix> -psf <psf file> -pdb <pdb file> -griddx <griddx file> ?options?"
  puts "Options:" 
  puts "  -gridpdb  -- pdb file for docking (default: -pdb)"
  puts "  -diel     -- dielectric constant (default: $defaultDiel; 1 with -pbc or -gbis)" 
  puts "  -temp     -- temperature in Kelvin (default: $defaultTemp)" 
  puts "  -ftemp    -- final temperature (default: $defaultTemp)" 
  puts "  -gscale   -- scaling factor for the grid (default: $defaultGScale)" 
  puts "  -extrab   -- extrabonds file (default: none)" 
  puts "  -conspdb  -- pdb file with constrained atoms (default: none)"
  puts "  -conscol  -- force constant column in conspdb (default: beta)"
  puts "  -fixpdb   -- pdb file with fixed atoms (default: none)"
  puts "  -fixcol   -- column in fixpdb (default: occupancy)"
  puts "  -scal14   -- 1-4 scaling (default: $defaultScaling1_4)"
  puts "  -step     -- docking protocol step (default: 1)" 
  #puts "  -parfiles -- parameter file list (default $defaultParFile)"
  puts "  -parfiles -- parameter file list"
  puts "  -minsteps -- number of minimization steps (default $defaultMinimize)"
  puts "  -numsteps -- number of time steps to run (default: $defaultNumSteps)" 
  puts "  -margin   -- extra length in patch dimension during simulation (default: $defaultMargin)"
  puts "  -pbc      -- use periodic boundary conditions (for explicit solvent)"
  puts "  -gbis     -- use generalized Born implicit solvent (not compatible with -pbc)"
  puts "  -dir      -- Working Directory (default: $defaultDir)"
  puts "  --lite    -- use gridforcelite, a faster but less accurate calculation of mdff forces"
#IMD options
  puts "  --imd     -- turn on Interactive Molecular Dynamics (IMD)"
  puts "  -imdport  -- port for IMD connection"
  puts "  -imdfreq  -- timesteps between sending IMD coordinates"
  puts "  --imdwait -- wait for IMD connection"
  puts "  --imdignore -- ignore steering forces from VMD" 
#xMDFF options only!
  puts "  --xmdff   -- set up xMDFF simulation.  The following options apply to xMDFF only."  
  puts "  -refs     -- reflection data file (mtz or cif). Required for xMDFF"  
  puts "  -refsteps -- number of refinement steps between map generation (default: $defaultRefSteps)"  
  puts "  -crystpdb -- text file (can be PDB) with PDB formatted CRYST line to supply symmetry information (default: none, but recommended)"  
  puts "  --mask    -- clean generated maps by applying a binary mask around structure (default: off)" 
  puts "  -mask_res -- resolution of mask density in Angstroms (default: $defaultMaskRes)" 
  puts "  -mask_cutoff -- cutoff distance of mask density in Angstroms (default: $defaultMaskCutoff)" 
  puts "  --bfs     -- calculate beta factors during every map generation step (useful for beta factor sharpening) (default: off)"
}

proc ::MDFF::Setup::mdff_setup { args } {

  variable defaultDiel 
  variable defaultScaling1_4
  variable defaultGScale 
  variable defaultTemp 
  variable defaultParFile
  variable defaultNumSteps 
  variable defaultMinimize
  variable defaultConsCol 
  variable defaultFixCol
  variable namdTemplateFile
  variable xMDFFTemplateFile
  variable xMDFFScriptFile
  variable defaultMargin 
  variable defaultDir
  variable defaultLite

  #xMDFF related variables
  variable defaultxMDFF   
  variable defaultBFS      
  variable defaultMask     
  variable defaultRefSteps
  variable defaultCrystPDB
  variable defaultMaskRes
  variable defaultMaskCutoff 
  
  #IMD related variables 
  variable defaultIMD
  variable defaultIMDPort
  variable defaultIMDFreq
  variable defaultIMDWait
  variable defaultIMDIgnore
  
  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_setup_usage
    error ""
  }

  # Get NAMD template and parameter files
  ::MDFF::Setup::init_files

  # Periodic simulation?
  set pos [lsearch -exact $args {-pbc}]
  if { $pos != -1 } {
    set pbc 1
    set args [lreplace $args $pos $pos]
  } else {
    set pbc 0
  }

  # Using GB implicit solvent?
  set pos [lsearch -exact $args {-gbis}]
  if { $pos != -1 } {
    set gbison 1
    set args [lreplace $args $pos $pos]
  } else {
    set gbison 0
  }

  if {$pbc && $gbison} {
	mdff_setup_usage
	error "Use of -gbis is not compatible with -pbc."
  }
  
  # parse switches
#  foreach {name val} $args {
#    switch -- $name {
#      -o          { set arg(o)        $val }
#      -psf        { set arg(psf)      $val } 
#      -pdb        { set arg(pdb)      $val }
#      -gridpdb    { set arg(gridpdb)  $val }
#      -diel       { set arg(diel)     $val }
#      -scal14     { set arg(scal14)   $val }
#      -temp       { set arg(temp)     $val }
#      -ftemp      { set arg(ftemp)    $val }
#      -griddx     { set arg(griddx)   $val }
#      -gscale     { set arg(gscale)   $val }
#      -extrab     { set arg(extrab)   $val }
#      -conspdb    { set arg(conspdb)  $val }
#      -conscol    { set arg(conscol)  $val }
#      -fixpdb     { set arg(fixpdb)   $val }
#      -fixcol     { set arg(fixcol)   $val }
#      -step       { set arg(step)     $val }
#      -parfiles   { set arg(parfiles) $val }
#      -numsteps   { set arg(numsteps) $val }
#      -minsteps   { set arg(minsteps) $val }
#      -margin     { set arg(margin)   $val }
#      #begind xMDFF related options
#      -xmdff      { set arg(xmdff)    $val }
#      -refsteps   { set arg(refsteps) $val }
#      -mask       { set arg(mask)     $val }
#      -bfs        { set arg(bfs)      $val }
#      -refs       { set arg(refs)     $val }
#      -crystpdb   { set arg(crystpdb) $val }
#    }
#  }
   
  for {set i 0} {$i < [llength $args]} {incr i} {
    switch -- [lindex $args $i] {
      -o          { set arg(o)        [lindex $args [expr $i + 1]] }
      -psf        { set arg(psf)      [lindex $args [expr $i + 1]] } 
      -pdb        { set arg(pdb)      [lindex $args [expr $i + 1]] }
      -gridpdb    { set arg(gridpdb)  [lindex $args [expr $i + 1]] }
      -diel       { set arg(diel)     [lindex $args [expr $i + 1]] }
      -scal14     { set arg(scal14)   [lindex $args [expr $i + 1]] }
      -temp       { set arg(temp)     [lindex $args [expr $i + 1]] }
      -ftemp      { set arg(ftemp)    [lindex $args [expr $i + 1]] }
      -griddx     { set arg(griddx)   [lindex $args [expr $i + 1]] }
      -gscale     { set arg(gscale)   [lindex $args [expr $i + 1]] }
      -extrab     { set arg(extrab)   [lindex $args [expr $i + 1]] }
      -conspdb    { set arg(conspdb)  [lindex $args [expr $i + 1]] }
      -conscol    { set arg(conscol)  [lindex $args [expr $i + 1]] }
      -fixpdb     { set arg(fixpdb)   [lindex $args [expr $i + 1]] }
      -fixcol     { set arg(fixcol)   [lindex $args [expr $i + 1]] }
      -step       { set arg(step)     [lindex $args [expr $i + 1]] }
      -parfiles   { set arg(parfiles) [lindex $args [expr $i + 1]] }
      -numsteps   { set arg(numsteps) [lindex $args [expr $i + 1]] }
      -minsteps   { set arg(minsteps) [lindex $args [expr $i + 1]] }
      -margin     { set arg(margin)   [lindex $args [expr $i + 1]] }
      -dir        { set arg(dir)      [lindex $args [expr $i + 1]] } 
      --lite      { set arg(lite)    1 }
      #begind xMDFF related options
      --xmdff     { set arg(xmdff)    1 }
      -refsteps   { set arg(refsteps) [lindex $args [expr $i + 1]] }
      --mask      { set arg(mask)     1 }
      --bfs       { set arg(bfs)      1 }
      -refs       { set arg(refs)     [lindex $args [expr $i + 1]] }
      -crystpdb   { set arg(crystpdb) [lindex $args [expr $i + 1]] }
      -mask_res   { set arg(maskres) [lindex $args [expr $i + 1]] }
      -mask_cutoff   { set arg(maskcutoff) [lindex $args [expr $i + 1]] }
      #begin IMD related options
      --imd       { set arg(imd)     1 }
      -imdport    { set arg(imdport)  [lindex $args [expr $i + 1]] }
      -imdfreq    { set arg(imdfreq)  [lindex $args [expr $i + 1]] }
      --imdwait   { set arg(imdwait) "yes" }
      --imdignore { set arg(imdignore) "yes" }
    }
  }
    
 
  if { [info exists arg(o)] } {
    set outprefix $arg(o)
  } else {
    mdff_setup_usage
    error "Missing output files prefix."
  }
  
  if { [info exists arg(psf)] } {
    set psf $arg(psf)
  } else {
    mdff_setup_usage
    error "Missing psf file."
  }
  
  if { [info exists arg(pdb)] } {
    set pdb $arg(pdb)
  } else {
    mdff_setup_usage
    error "Missing pdb file."
  }

  if { [info exists arg(diel)] } {
    set diel $arg(diel)
  } elseif {$pbc || $gbison} {
    set diel 1
  } else {
    set diel $defaultDiel
  }

  if { [info exists arg(scal14)] } {
    set scal14 $arg(scal14)
  } else {
    set scal14 $defaultScaling1_4
  }

  if { [info exists arg(temp)] } {
    set itemp $arg(temp)
  } else {
    set itemp $defaultTemp
  }

  if { [info exists arg(ftemp)] } {
    set ftemp $arg(ftemp)
  } else {
    set ftemp $itemp
  }

  if { [info exists arg(parfiles)] } {
    set parfiles $arg(parfiles)
  } else {
    file copy -force $defaultParFile .
    set parfiles [file tail $defaultParFile]
  }

  if { [info exists arg(numsteps)] } {
    set numsteps $arg(numsteps)
  } else {
    set numsteps $defaultNumSteps
  }


  if { [info exists arg(minsteps)] } {
    set minsteps $arg(minsteps)
  } else {
    set minsteps $defaultMinimize
  }

  if { [info exists arg(griddx)] } {
    set grid $arg(griddx) 
  } else {
    mdff_setup_usage 
    error "Missing grid dx file name."
  }
  
  if { [info exists arg(gscale)] } {
    set gscale $arg(gscale)
  } else {
    set gscale $defaultGScale
  }
  
  if { [info exists arg(extrab)] } {
    set extrab $arg(extrab)
  } else {
    set extrab 0
  }
  
  if { [info exists arg(gridpdb)] } {
    set gridpdb $arg(gridpdb)
  } else {
    set gridpdb $pdb
  }

  if { [info exists arg(conspdb)] } {
    set conspdb $arg(conspdb)
  } else {
    set conspdb 0
  }

  if { [info exists arg(conscol)] } {
    set conscol $arg(conscol)
  } else {
    set conscol $defaultConsCol
  }

  if { [info exists arg(fixpdb)] } {
    set fixpdb $arg(fixpdb)
  } else {
    set fixpdb 0
  }

  if { [info exists arg(fixcol)] } {
    set fixcol $arg(fixcol)
  } else {
    set fixcol $defaultFixCol
  }

  if { [info exists arg(minsteps)] } {
    set minsteps $arg(minsteps)
  } else {
    set minsteps $defaultMinimize
  }

  if { [info exists arg(margin)] } {
    set margin $arg(margin)
  } else {
    set margin $defaultMargin
  }
  
  if { [info exists arg(dir)] } {
    set dir $arg(dir)
  } else {
    set dir $defaultDir
  }
  
  if { [info exists arg(lite)] } {
    set lite $arg(lite)
  } else {
    set lite $defaultLite
  }

  if { [info exists arg(step)] } {
    set step $arg(step)
  } else {
    # puts "No step number was specified. Assuming step 1.."
    set step 1
  }
  
  puts "starting xmdff section"
  if { [info exists arg(xmdff)] } {
    set xmdff $arg(xmdff)
  } else {
    set xmdff $defaultxMDFF
  }
  if { $xmdff } {
    if { [info exists arg(refs)] } {
      set refs $arg(refs)
    } else {
      mdff_setup_usage
      error "Missing structure factors file."
    }

    if { [info exists arg(mask)] } {
      set mask $arg(mask)
    } else {
      set mask $defaultMask
    }

    if { [info exists arg(bfs)] } {
      set bfs $arg(bfs)
    } else {
      set bfs $defaultBFS
    }
  
    if { [info exists arg(crystpdb)] } {
      set crystpdb $arg(crystpdb)
    } else {
      set crystpdb $defaultCrystPDB
    }

    if { [info exists arg(refsteps)] } {
      set refsteps $arg(refsteps)
    } else {
      set refsteps $defaultRefSteps
    }
    
    if { [info exists arg(maskres)] } {
      set maskres $arg(maskres)
    } else {
      set maskres $defaultMaskRes
    }
    
    if { [info exists arg(maskcutoff)] } {
      set maskcutoff $arg(maskcutoff)
    } else {
      set maskcutoff $defaultMaskCutoff
    }
  }
  puts "starting IMD section"
  if { [info exists arg(imd)] } {
    set imd $arg(imd)
  } else {
    set imd $defaultIMD
  }
  
  if { $imd } {
    if { [info exists arg(imdport)] } {
      set imdport $arg(imdport)
    } else {
      set imdport $defaultIMDPort
    }

    if { [info exists arg(imdfreq)] } {
      set imdfreq $arg(imdfreq)
    } else {
      set imdfreq $defaultIMDFreq
    }

    if { [info exists arg(imdwait)] } {
      set imdwait $arg(imdwait)
    } else {
      set imdwait $defaultIMDWait
    }
  
    if { [info exists arg(imdignore)] } {
      set imdignore $arg(imdignore)
    } else {
      set imdignore $defaultIMDIgnore
    }
  }

  if {$xmdff} {
    file copy -force $xMDFFTemplateFile $dir
    file copy -force $xMDFFScriptFile $dir
    file delete "maps.params"
    if [catch {exec phenix.maps} result] {
      puts $result
    } else {
      set frpdb [open "maps.params" "r"]
      set spdb [read $frpdb]
      close $frpdb
      set fwpdb [open "maps.params" "w"]
      
      regsub "pdb_file_name = None" $spdb "pdb_file_name = mapinput.pdb" spdb
      regsub "file_name = None" $spdb "file_name = $refs" spdb
      regsub -all "exclude_free_r_reflections = False" $spdb "exclude_free_r_reflections = True" spdb
      puts $fwpdb $spdb
      close $fwpdb

      file rename -force "maps.params" $dir
    }

  } else {
    # Copy NAMD template file to working directory
    file copy -force $namdTemplateFile $dir
  }
  set outname [file join $dir ${outprefix}-step${step}]
  puts "mdff) Writing NAMD configuration file ${outname}.namd ..."
  
  set out [open ${outname}.namd w]     

  puts $out "###  Docking -- Step $step" 
  puts $out " "   
  puts $out "set PSFFILE $psf"        
  puts $out "set PDBFILE $pdb"
  puts $out "set GRIDPDB $gridpdb"
  puts $out "set GBISON $gbison"
  puts $out "set DIEL $diel"        
  puts $out "set SCALING_1_4 $scal14"
  puts $out "set ITEMP $itemp"   
  puts $out "set FTEMP $ftemp"   
  puts $out "set GRIDFILE $grid"   
  puts $out "set GSCALE $gscale"   
  puts $out "set EXTRAB [list $extrab]"   
  puts $out "set CONSPDB $conspdb"
  if {$conspdb != "0" } {
    puts $out "set CONSCOL $conscol"
  }
  puts $out "set FIXPDB  $fixpdb"
  if {$fixpdb != "0" } {
    puts $out "set FIXCOL $fixcol"
  }
  
  if {$xmdff} {
    puts $out "set REFINESTEP $refsteps"
    puts $out "set REFS $refs"
    puts $out "set BFS $bfs"
    puts $out "set MASK $mask"
    puts $out "set MASKRES $maskres"
    puts $out "set MASKCUTOFF $maskcutoff"
    puts $out "set CRYSTPDB $crystpdb"
  }
  puts $out " " 
  
  if {$step >  1 } {
    set prevstep [expr $step - 1]
    set inputname "${outprefix}-step${prevstep}"
    set prevnamd "${inputname}.namd"
    if { ![file exists $prevnamd] } {
      puts "Warning: Previous NAMD configuration file $prevnamd not found." 
      puts "You may need to manually edit the variable INPUTNAME in the file ${outname}.namd."
    }
    puts $out "set INPUTNAME $inputname"  
  }

  puts $out "set OUTPUTNAME ${outprefix}-step${step}"
  puts $out " "
  puts $out "set TS $numsteps"
  puts $out "set MS $minsteps"
  puts $out " "
  puts $out "set MARGIN $margin"
  puts $out " "
  puts $out "####################################"
  puts $out " "
  puts $out "structure \$PSFFILE"
  puts $out "coordinates \$PDBFILE"
  puts $out " "
  puts $out "paraTypeCharmm on"
  foreach par $parfiles {
    puts $out "parameters $par"
  }
  if $pbc {
    puts $out ""
    puts $out "if {\[info exists INPUTNAME\]} {"
    puts $out "  BinVelocities \$INPUTNAME.restart.vel"
    puts $out "  BinCoordinates \$INPUTNAME.restart.coor"
    puts $out "  ExtendedSystem \$INPUTNAME.restart.xsc"
    puts $out "} else {"
    puts $out "  temperature \$ITEMP"
    ::MDFF::Setup::get_cell $psf $pdb $out
    puts $out "}"

    puts $out "PME yes"
    puts $out "PMEGridSpacing 1.0"
    puts $out "PMEPencils 1"

    puts $out "wrapAll on"

  } else {
    puts $out ""
    puts $out "if {\[info exists INPUTNAME\]} {"
    puts $out "  BinVelocities \$INPUTNAME.restart.vel"
    puts $out "  BinCoordinates \$INPUTNAME.restart.coor"
    puts $out "} else {"
    puts $out "  temperature \$ITEMP"
    puts $out "}"

  }
  puts $out " "
  if {$imd} {
    puts $out "IMDon on"
    puts $out "IMDport $imdport"
    if {$imdfreq > 0} {
      puts $out "IMDfreq $imdfreq"
    }
    puts $out "IMDwait $imdwait"
    puts $out "IMDignore $imdignore"
  }
  puts $out " "
  if {$lite} {
    puts $out "gridforcelite on"
    puts $out " "
  }
  if {$xmdff} {
    puts $out "source [file tail $xMDFFTemplateFile]"
  } else {
    puts $out "source [file tail $namdTemplateFile]"
  }
  if $xmdff {
    puts $out "#BEGIN XMDFF NECESSARY FUNCTIONS"
    puts $out "if {\[info exists INPUTNAME\]} {"
    puts $out "  exec vmd -dispdev text -e xmdff_phenix.tcl -args \$PSFFILE \$INPUTNAME \$GRIDFILE \$REFS \$BFS \$MASK \$CRYSTPDB \$MASKRES \$MASKCUTOFF > map.log"
    puts $out "} else {"
    puts $out "  exec vmd -dispdev text -e xmdff_phenix.tcl -args \$PSFFILE \$PDBFILE \$GRIDFILE \$REFS \$BFS \$MASK \$CRYSTPDB \$MASKRES \$MASKCUTOFF > map.log"
    puts $out "}"
    puts $out "if {\$MS != 0} {"
    puts $out "  minimize \$MS"
    puts $out "  reinitvels \$ITEMP"
    puts $out "}"
    puts $out "if {\$ITEMP != \$FTEMP} {"
    puts $out "  set ANNEALSTEP \[expr abs(\$FTEMP-\$ITEMP)*100\]"
    puts $out "  run \$ANNEALSTEP"
    puts $out "  exec vmd -dispdev text -e xmdff_phenix.tcl -args \$PSFFILE \$OUTPUTNAME \$GRIDFILE \$REFS \$BFS \$MASK \$CRYSTPDB \$MASKRES \$MASKCUTOFF > map.log"
    puts $out "  reloadGridforceGrid"
    puts $out "}"
    puts $out "for {set i 0} {\$i < \$TS/$\REFINESTEP} {incr i} {"
    puts $out "  run \$REFINESTEP"
    puts $out "  exec vmd -dispdev text -e xmdff_phenix.tcl -args \$PSFFILE \$OUTPUTNAME \$GRIDFILE \$REFS \$BFS \$MASK \$CRYSTPDB \$MASKRES \$MASKCUTOFF > map.log "
    puts $out "  reloadGridforceGrid"
    puts $out " }"
  }
  close $out

}

proc ::MDFF::Setup::get_cell {psf pdb out} {
  set molid [mol new $psf type psf waitfor all]
  mol addfile $pdb type pdb waitfor all

  set sel [atomselect $molid {noh water}]

  if { [$sel num] == 0 } {
    $sel delete
    mol delete $molid
    error "Could not determine the periodic cell information. No water molecules were found in the input structure."
  }
  set minmax [measure minmax $sel]
  set vec [vecsub [lindex $minmax 1] [lindex $minmax 0]]
  puts $out "  cellBasisVector1 [lindex $vec 0] 0 0"
  puts $out "  cellBasisVector2 0 [lindex $vec 1] 0"
  puts $out "  cellBasisVector3 0 0 [lindex $vec 2]"
  set center [measure center $sel]
  puts $out "  cellOrigin $center"
  $sel delete
  
  mol delete $molid

}


proc ::MDFF::Setup::mdff_gridpdb_usage { } {
 
  variable defaultGridpdbSel

  puts "Usage: mdff gridpdb -psf <input psf> -pdb <input pdb> -o <output pdb> ?options?"
  puts "Options:" 
  puts "  -seltext   -- atom selection text  (default: $defaultGridpdbSel)"
}

proc ::MDFF::Setup::mdff_gridpdb { args } {

  variable defaultGridpdbSel

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_gridpdb_usage
    error ""
  }

  # parse switches
  foreach {name val} $args {
    switch -- $name {
      -psf        { set arg(psf)      $val }
      -pdb        { set arg(pdb)      $val }
      -o          { set arg(o)        $val }
      -seltext    { set arg(seltext)  $val }
    }
  }
    
  if { [info exists arg(o)] } {
    set gridpdb $arg(o)
  } else {
    mdff_gridpdb_usage
    error "Missing output gridpdb file name."
  }
  
  if { [info exists arg(pdb)] } {
    set pdb $arg(pdb)
  } else {
    mdff_gridpdb_usage
    error "Missing pdb file."
  }

  if { [info exists arg(psf)] } {
    set psf $arg(psf)
  } else {
    mdff_gridpdb_usage
    error "Missing psf file."
  }

  if { [info exists arg(seltext)]} {
    set seltext $arg(seltext)
  } else {
    set seltext $defaultGridpdbSel
  }
  
  set molid [mol new $psf type psf waitfor all]
  mol addfile $pdb type pdb waitfor all
  set all [atomselect $molid all]
  $all set occupancy 0

  if { $seltext == "all" } {
    $all set beta [$all get mass]
    $all set occupancy 1
  } else {
    $all set beta 0
    set sel [atomselect $molid $seltext]
    if {[$sel num] == 0} {
      error "empty atomselection"
    } else {
      $sel set occupancy 1
      $sel set beta [$sel get mass]
    }  
    $sel delete
  }

  $all writepdb $gridpdb
  $all delete
  
  return 

}

proc ::MDFF::Setup::init_files {} {
  global env
  variable defaultParFile
  variable namdTemplateFile
  variable xMDFFTemplateFile
  variable xMDFFScriptFile
  set defaultParFile [file join $env(CHARMMPARDIR) par_all27_prot_lipid_na.inp]
  set namdTemplateFile [file join $env(MDFFDIR) mdff_template.namd]
  set xMDFFTemplateFile [file join $env(MDFFDIR) xmdff_template.namd]
  set xMDFFScriptFile [file join $env(MDFFDIR) xmdff_phenix.tcl]
}

proc ::MDFF::Setup::mdff_constrain_usage { } {

  variable defaultK
  variable defaultConsCol
 
  puts "Usage: mdff constrain <atomselection> -o <pdb file> ?options?"
  puts "Options:"
  puts "  -col <column> (default: $defaultConsCol)"
  puts "  -k <force constant in kcal/mol/A^2> (default: $defaultK)"
  
}

proc ::MDFF::Setup::mdff_fix_usage { } {

  variable defaultFixCol
 
  puts "Usage: mdff fix <atomselection> -o <pdb file> ?options?"
  puts "Options:"
  puts "  -col <column> (default: $defaultFixCol)"
  
}

proc ::MDFF::Setup::mdff_constrain { args } {

  variable defaultK
  variable defaultConsCol

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_constrain_usage
    error ""
  }

  set sel [lindex $args 0]
  if { [$sel num] == 0 } {
    error "empty atomselection"
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -o     { set arg(o)     $val }
      -col   { set arg(col)   $val }
      -k     { set arg(k)     $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if { [info exists arg(o)] } {
    set outputFile $arg(o)
  } else {
    mdff_constrain_usage
    error "Missing output pdb file."
  }

  if { [info exists arg(col)] } {
    set col $arg(col)
  } else {
    set col $defaultConsCol
  }

  if { $col == "beta" || $col == "B" } {
    set col "beta"
  } elseif { $col == "occupancy" || $col == "O" } {
    set col "occupancy"
  } elseif { $col == "x" || $col == "X" } {
    set col "x"
  } elseif { $col == "y" || $col == "Y" } {
    set col "y"
  } elseif { $col == "z" || $col == "Z" } {
    set col "z"
  } else {
    error "Unrecognized column."
  }

  if { [info exists arg(k)] } {
    set k $arg(k)
  } else {
    set k $defaultK
  }

  set molid [$sel molid]
  set all [atomselect $molid all]
  set bakCol [$all get $col]
  $all set $col 0
  $sel set $col $k
  $all writepdb $outputFile
  $all set $col $bakCol
  $all delete

  return

}

proc ::MDFF::Setup::mdff_fix { args } {

  variable defaultFixCol

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_fix_usage
    error ""
  }

  set sel [lindex [lindex $args 0] 0]
  if { [$sel num] == 0 } {
    error "mdff_constrain: empty atomselection."
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -o     { set arg(o)     $val }
      -col   { set arg(col)   $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if { [info exists arg(o)] } {
    set outputFile $arg(o)
  } else {
    mdff_fix_usage
    error "Missing output pdb file."
  }

  if { [info exists arg(col)] } {
    set col $arg(col)
  } else {
    set col $defaultFixCol
  }

  if { $col == "beta" || $col == "B" } {
    set col "beta"
  } elseif { $col == "occupancy" || $col == "O" } {
    set col "occupancy"
  } elseif { $col == "x" || $col == "X" } {
    set col "x"
  } elseif { $col == "y" || $col == "Y" } {
    set col "y"
  } elseif { $col == "z" || $col == "Z" } {
    set col "z"
  } else {
    error "Unrecognized column."
  }

  set molid [$sel molid]
  set all [atomselect $molid all]
  set bakCol [$all get $col]
  $all set $col 0
  $sel set $col 1
  $all writepdb $outputFile
  $all set $col $bakCol
  $all delete

  return

}
