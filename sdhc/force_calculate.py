#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Kimmo Sääskilahti
ksaaskil@gmail.com

Made some modifications by Liang Ting
liangting.zj@gmail.com
2020/2/7 14:22:15
"""

import numpy as np
import sys

__all__ = ["fcCalc"]

class fcCalc(object):
    """
    Class for computing force constants between atoms.
    Uses the `Python library interface <https://lammps.sandia.gov/doc/Python_library.html>`_
    of LAMMPS so you need to have (1) `lammps` in your `PYTHONPATH` and
    (2) `liblammps.so` available for the Python package.
    :param fileprefix: File prefix (TODO What is this)
    :type fileprefix: str
    :param restartfile: LAMMPS restart file (The restart file is also required, but here not as an attribute)  
    :type restartfile: str
    """
    def __init__(self, fileprefix):
    	
        self.fileprefix = fileprefix
        self.Kij = None
        self.inds_left = None                  ## Used to find the index of the atoms on the left
        self.inds_right = None                 ## Used to find the index of the atoms on the right
        self.inds_interface = None             ## Used to compare with the atomic index in the dump_velovity file
        self.ids_L = None                      ## Used to index the speed ordinal (left)
        self.ids_R = None                      ## Used to index the speed ordinal (right)
        self.natoms = None
        self.lmp = None                        ## Python-Lammps interface

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def preparelammps(self, in_lammps = None, w_interface = 6.0, show_log = False):
        """
        Prepare the LAMMPS object for computing force constants.
        :param fileprefix: File prefix (lammps in_file)
        :type fileprefix: str
        :param w_interface: Width of the area of atoms to include in the interface, defaults to 6.0
        :type w_interface: float, optional
        :param show_log: Whether to write the log file to the generate_force.log file, if true, no log file will be generated. (default False)
        :type show_log: boolean, optional
        :return: None
        """
        from lammps import lammps
        
        cmd_list = ['-log', 'generate_force.log', '-screen', 'none']
        
        if not show_log:
           cmd_list += ['-echo', 'none']
           
        self.lmp = lammps(cmdargs = cmd_list)

        if in_lammps is not None:
        	 pass
        else:
        	 sys.exit('\n\tFile error: The in file that lammps needs to read does not exist!\n')

        #lines = open(in_lammps,'r').readlines()
        #for line in lines: self.lmp.command(line) 
        """
        Do the same as lmp.command(filename) but don't allow the script to be
        continued after quit.
        """
        lines = open(in_lammps,'r').readlines()
        for line in lines:
           #print(line)
           if "quit" in line and line[0] != "#":
              sys.exit('\n\tQuit error: It is not appropriate '
                       'to exit the execution of the in.file at this time\n')    ## Quit lammps if find the keyword 'quit'
           else:
              self.lmp.command(line)
             
        # Get the position of the interface, at the middle by default (0.5)
        
        ylo = self.lmp.extract_global("boxylo", 1)
        yhi = self.lmp.extract_global("boxyhi", 1)
        #print "Box is [%f,%f]." % (ylo, yhi)                            # Print the boundaries of the box (optional)
        
        # These variables (including middle, mid_left, and mid_right) must be predefined in the in_file
        middle = self.lmp.extract_variable("middle", "all", 0)           
        mid_left = self.lmp.extract_variable("mid_left", "all", 0)
        mid_right = self.lmp.extract_variable("mid_right", "all", 0)     # group = group ID ('all') (ignored for equal-style variables)
        w_interface_infile = (mid_right - mid_left)/2

        if (w_interface_infile != w_interface):
           sys.exit('\n\t Parameter setting ERROR: The settings for w_interface do ' 
           'not match those in lammps\n')

        # Coordinates ordered by atom ID
        coords_data = self.lmp.gather_atoms("x", 1, 3)
        
        # Coordinates in a numpy array
        coords = np.array(coords_data[:], dtype=np.dtype('f8'))
        self.natoms = self.lmp.extract_global("natoms", 0)
        
        '''  Data structures after rashape
        
         atom1_x (coords_data[0])        atom1_y (coords_data[1])         atom1_z (coords_data[2])        
         atom2_x (coords_data[3])        atom2_y (coords_data[4])         atom2_z (coords_data[5]) 
        '''
        coords = np.reshape(coords, (self.natoms, 3))
        
        # Y-coordinates in a Numpy array
        ys = coords[:, 1]	                                                 ## xs=coords[:, 0]	,  ys = coords[:, 1],  zs = coords[:, 2]		
         
        '''
        (1) minimize            0    1.0e-3   1000   1000     # (after relaxing the structures to the potential minimum)
            run                 0
            After the structure is completely relaxed, the process of energy minimization must be added
            
        (2) Since energy minimization may result in different atomic Numbers on the left and right sides of the interface, 
            the definition of the width of the interface can be fine-tuned.
            
        (3) The program will automatically identify if the left and right atom ids are the same as the atom id of the dump_velcity_file.
        '''  
        # Atom on the left side?
        mask_left = np.logical_and(ys < middle, ys >= mid_left-1)          ## return boolean
        # Atom on the right side?
        mask_right = np.logical_and(ys > middle, ys <= mid_right)

        # Note that these indices differ from atom IDs by a factor of one
        
        self.inds_left = np.where(mask_left)[0]
        self.inds_right = np.where(mask_right)[0]     
       
        # All atom indices sorted by atom ID, duplicates removed
        # Note that these indices differ from atom IDs by a factor of one 
        
        self.inds_interface = np.unique(np.concatenate((self.inds_left, self.inds_right))) 
        
        # Where are the atoms of the left atom set

        self.ids_L = np.in1d(self.inds_interface, self.inds_left)     # To find the subset
        self.ids_L = np.where(self.ids_L)[0]                     # or np.where(np.in1d(aArray,bArray))[0]
        
        # Atoms of the right set

        self.ids_R = np.in1d(self.inds_interface, self.inds_right)
        self.ids_R = np.where(self.ids_R)[0]
        
        # Get information from the in file 
        
        N = self.lmp.extract_variable("N", "all", 0)
        NL = self.lmp.extract_variable("NL", "all", 0)
        NR = self.lmp.extract_variable("NR", "all", 0)
        
        # Determine whether the number of left and right atoms is equal
        
        if (len(self.ids_L) != len(self.ids_R) or len(self.ids_L) != NR or len(self.ids_R) != NL):
            sys.exit('\n\tLAMMPS SETUP ERROR: Number of atoms on left and right side don\'t match! ' 
                         'Checking the width definition of the interface is an good option.\n')

    def fcCalc(self, hstep):
        """
         Compute force constants and store to `self.Kij`.
        :param hstep: Step to use in finite differences (the finite displacement method)
        :type hstep: float
        :return: None
        
        This function reads in the [dFi/duj] force constanst from 
        LAMMPS compute commands. The structure of the infile should be blocks
        of data that are N atoms long. Here, N means the number of atoms in
        the interface region. Each atom of the N atoms in the block 
        is the 'i' index in d^2Fi/duj. The Python_LAMMPS code loops over all atoms j
        on the left side of the interface; j in the dFi/duj expression. There
        are NL = NR atoms on the left side of the inteface, i.e. the number on 
        each side should (and MUST) be the same. Each atom on the left side, j, is 
        looped over; in each loop iteration the atom is moved in the +x, then -x, 
        then back to equilibrium. Then the atom is moved in the +y, then -y, then
        back to equilibrium. Finally, the atom is moved in the +z, then -z, then 
        back to equilibrium.
    
        Thus, each line in each block in the file corresponds to an atom i. 
        Each element in each line is the force felt by the atom in the x, y, and 
        z direction (the force vector elements). For each block, the atom j is 
        moved iteratively in the +x, then-x, then +y ... etc.
    
        The function returns kij, the matrix elements of the force constants 
        between atoms on either side of the interface.
        kij has the shape [3*nr,3*nl] -> [dFi-x,y,z,duj-x,y,z]. Each individual 
        force constant is defined as:
        
             dFi_a
           ----------
             drj_b
        
        Which is interpreted as the change in force on atom i in the direction
        a = (x,y,z) due to the movement of atom j in b=(x,y,z).
        Each element of first dimension of the kij matrix corresponds to the 
        force on atom i due to the movement of atom j in each direction 
        
         dF1_x            dF1_x            dF1_x           dF1_x        
        ----------       ----------       ----------     ----------  .....
         dr1_x            dr1_y            dr1_z           dr2_x
        
        
          dF1_y           dF1_y             dF1_y          dF1_y        
        ----------       ----------       ----------     ----------  .....
          dr1_x           dr1_y             dr1_z          dr2_x
        
        
          dF1_z           dF1_z             dF1_z          dF1_z        
        ----------       ----------       ----------     ----------  .....
          dr1_x           dr1_y             dr1_z          dr2_x
          
          
          dF2_x           dF2_x             dF2_x          dF2_x        
        ----------       ----------       ----------     ----------  .....
          dr1_x           dr1_y             dr1_z          dr2_x
        
            :               :                  :               :
            :               :                  :               :    
        """
        lmp = self.lmp
        natoms = self.natoms
        inds_left = self.inds_left
        inds_right = self.inds_right

        # One-dimensional indices of the atoms on the right side (Y-coordinates)
        
        inds_right_1d = np.concatenate((3 * inds_right, 3 * inds_right + 1, 3 * inds_right + 2))
        inds_right_1d = np.sort(inds_right_1d)

        Kij = np.zeros((len(inds_left) * 3, len(inds_right) * 3))

        # Loop over the atoms on the left side
        
        print ('\nNow start the loop and move the atoms to calculate the force constant!!!')
        for i1 in range(0, len(inds_left)):
            #  (For test) for i1 in range(0,10):
            # Index of the atom on the left
            ind1 = inds_left[i1]
            # Find the indices of atom ind1 in the 1D array (For test)
            indx = 3 * ind1                                    # directions x
            indy = 3 * ind1 + 1                                # directions y
            indz = 3 * ind1 + 2                                # directions z

            print "\tMoving atom %i/%i." % (i1 + 1, len(inds_left))
            
            # Move atom to directions x(0), y(1), and z(2)
            
            for direction in [0, 1, 2]:
                # Index of the displaced degree of freedom
                index = 3 * ind1 + direction
                # Get the coordinates from LAMMPS
                yc = lmp.gather_atoms("x", 1, 3)
                # Move the atom
                yc[index] += hstep
                # Communicate to LAMMPS
                lmp.scatter_atoms("x", 1, 3, yc)
                # Run LAMMPS to update the forces
                lmp.command("run 0 post no")         ## The 'post no' means the full timing summary is skipped; only a one-line summary timing is printed
                # Gather the forces
                fc1 = lmp.gather_atoms("f", 1, 3)    ## The force (including x, y, z) on all the atoms in the interface group
                # print "1=",fc1[0]
                # print type(fc1)
                fc1 = np.array(fc1, dtype=np.dtype('f8'))
                # print "2=",fc1[0]
                # print fc1[index]                   ## Print the force of the current atom (if you need to check)
                # Move to negative direction
                yc[index] -= 2 * hstep
                lmp.scatter_atoms("x", 1, 3, yc)
                lmp.command("run 0 post no")
                fc2 = lmp.gather_atoms("f", 1, 3)
                fc2 = np.array(fc2, dtype=np.dtype('f8'))
                
                # print fc2[index]                   ## Print the force (negative direction) of the current atom (if you need to check)
                # Fill one row of spring constant matrix
                
                '''Calculate the force on all atoms in 
                the interface due to the current atom's displacement'''
                
                Kij[3 * i1 + direction, :] = (fc2[inds_right_1d] - fc1[inds_right_1d]) / (2.0 * hstep)   ## fc2 - fc1 (Yes) ---- (Negative-Positive)
                yc[index] += hstep
                lmp.scatter_atoms("x", 1, 3, yc)       ## Back to equilibrium.

        self.Kij = Kij
        lmp.close()

    def writeToFile(self):
        '''
        Write `self.Kij` to files starting with `self.fileprefix`.
        :return: None
        '''
        np.save(self.fileprefix + '.Kij.npy', self.Kij)
        np.save(self.fileprefix + '.ids_L.npy', self.ids_L)
        np.save(self.fileprefix + '.ids_R.npy', self.ids_R)
        np.save(self.fileprefix + '.ids_Interface.npy', self.inds_interface)
        np.savetxt(self.fileprefix, self.Kij, delimiter=' ')
        print('\nKij have been written to the ' + self.fileprefix + ' file\n')


if __name__ == "__main__":
	
    # import argparse
    # parser=argparse.ArgumentParser()
    # parser.add_argument("filePrefix",help="The prefix of file for which to calculate the force constants")
    # parser.add_argument("hstep",default=0.01,help="The displacement used in the finite-difference evaluation of force constants.")

    # args=parser.parse_args()
    # fileprefix=args.filePrefix
    # hstep=args.hstep
    
    fileprefix = 'Fij.dat'
    hstep = 0.01                                     ## Maybe hstep = 0.01 is a good choice
    in_file = 'forces.in'                            ## The in file for lammps
    
    with fcCalc(fileprefix) as fc:
        fc.preparelammps(in_lammps = in_file, w_interface = 6.0)
        fc.fcCalc(hstep)
        fc.writeToFile()
        print('Forces Calculate All Done\n')
