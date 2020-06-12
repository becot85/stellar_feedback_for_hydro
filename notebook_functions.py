import numpy as np
from datetime import date

#########################
#    Get Ejecta Rate    #
#########################
def get_ejecta_rate(inst, i_t, elem_list):
    
    '''
    
    Return the rate [Msun/yr] of elements at a given timestep,
    as well as the rate of total mass ejected and total metal
    mass ejected.
    
    Arguments
    =========
    
        inst: SYGMA instance
        i_t: Timestep index in the SYGMA instance
        elem_list: List of elements to be considered
    
    '''
    
    # Check whether the elements are available
    if not elements_available(inst, elem_list):
        return None

    # Declare the mass ejecta array for the elements
    nb_elem = len(elem_list)
    m_ej_elements = np.zeros(nb_elem)

    # Declare the isotope index that are H, He, and Li
    H_He_Li_index = []
    
    # For each element provided for the function ..
    for i_elem in range(nb_elem):
        
        # For each isotope included in the yields table ..
        for i_iso in range(inst.nb_isotopes):
            
            # If that isotope is part of the considered element ..
            the_elem = inst.history.isotopes[i_iso].split("-")[0]
            if the_elem == elem_list[i_elem]:
                
                # Collect the mass ejected [Msun]
                m_ej_elements[i_elem] += inst.mdot[i_t][i_iso]

            # Collect list of non-metal indexes
            if the_elem in ["H", "He", "Li"]:
                H_He_Li_index.append(i_iso)

    # Total mass ejected, including all elements in the
    # SYGMA instance (not only the one in elem_list)
    m_ej_tot = sum(inst.mdot[i_t])

    # Total metal mass ejected, including all elements in the
    # SYGMA instance (not only the one in elem_list)
    m_ej_Z = 0.0
    for i_iso in range(inst.nb_isotopes):
        if i_iso not in H_He_Li_index:
            m_ej_Z += inst.mdot[i_t][i_iso]
    
    # Return the enrichment rate [Msun/yr]
    return m_ej_elements / inst.history.timesteps[i_t],\
           m_ej_tot / inst.history.timesteps[i_t],\
           m_ej_Z / inst.history.timesteps[i_t]
    

#########################
#    Get Energy Rate    #
#########################
def get_energy_rate(inst, i_t, e_per_CC_SN, e_per_SN_Ia):
    
    '''
    
    Return the energy rate [erg/s] of elements at a given timestep,
    as well as the rate of total mass ejected and total metal
    mass ejected.
    
    Arguments
    =========
    
        inst: SYGMA instance
        i_t: Timestep index in the SYGMA instance
        e_per_CC_SN: Energy [erg] per core-collapse SN
        e_per_SN_Ia: Energy [erg] per Type Ia SN
    
    '''

    # Total amount of energy release during timestep i_t [erg]
    e_CC = inst.sn2_numbers[i_t] * e_per_CC_SN
    e_Ia = inst.sn1a_numbers[i_t] * e_per_SN_Ia

    # Convert the duration of the current timestep from yr to [s]
    dt_in_sec = inst.history.timesteps[i_t] * 3.154e+7

    # Return the kinetic energy rates
    return e_CC / dt_in_sec,\
           e_Ia / dt_in_sec


############################
#    Elements Available    #
############################
def elements_available(inst, elem_list):

    '''
    
    Check whether the elements provided in a list
    are available in a NuPyCEE GCE instance.
    
    Arguments
    =========
    
        inst: NuPyCEE instance (SYGMA or OMEGA)
        elem_list: List of elements
    
    '''

    # Define the list of un-available elements
    not_available = []
    
    # For each element ..
    for elem in elem_list:
    
        # Add the element to the list if not in the GCE instance
        if not elem in inst.history.elements:
            not_available.append(elem)

    # Print un-available elements if any
    if len(not_available) > 0:
        print("Error - The following elements are not available:")
        print("  ",not_available)
        return False
        
    # Return True if everything is ok
    return True
    

#########################
#    Fill With Space    #
#########################
def fill_with_space(thing, spacing=14):

    '''
    
    Take something (number or string) and convert it
    into a string with extra spaces at the end.
    
    Arguments
    =========
    
        thing: Number or string where spaces will be added
        spacing: Total lenght of the output string with spaces

    '''
    
    # Make sure the thing is a string
    thing_str = str(thing)
    
    # Number of spaces (" ") to be added
    nb_space = spacing - len(thing_str)
    
    # Add spaces
    for i in range(nb_space):
        thing_str += " "
        
    # Return the string with spaces
    return thing_str


###########################
#    Write Main Header    #
###########################
def write_main_header(enzo_table, prepared_by, params):

    '''

    Write the table header that only appears once at
    the top of the Enzo feedback table

    Arguments
    =========

        enzo_table: File where the header is beeing written
        prepared_by: Name(s) of the person(s) creating the file
        params: List of SYGMA parameters that have been specified
                Those exclude all parameters that have been left 
                to their default value

    '''

    # Write general information
    enzo_table.write("H Chemical enrichment feedback from SYGMA (NuPyCEE)\n")
    enzo_table.write("H \n")
    enzo_table.write("H Prepared by: " + prepared_by + "\n")
    enzo_table.write("H Date: " + str(date.today()) + "\n")
    enzo_table.write("H \n")
    enzo_table.write("H Ejecta are provided in the form of rate [Msun/yr]\n")
    enzo_table.write("H Kinetic energies are provided in the form of rate [erg/s]\n")
    enzo_table.write("H \n")

    # Write the list of parameters
    enzo_table.write("H List of SYGMA parameters\n")
    for key in params.keys():
        enzo_table.write("H "+key+": "+str(params[key])+"\n")


##################################
#    Write Metallicity Header    #
##################################
def write_metallicity_header(enzo_table, inst, elements, spacing=14):

    '''

    Write the table metallicity header that shows the
    metallicity as well as the columns identification

    Arguments
    =========

        enzo_table: File where the header is beeing written
        inst: SYGMA instance (for a given metallicity)
        elements: List of elements considered
        spacing: Total string length of each table column

    '''

    # Add space in between metallicity entries
    enzo_table.write("\n")

    # Write metallicity
    enzo_table.write("H Z="+str(inst.iniZ)+"\n")

    # Write the time column
    enzo_table.write("H "+fill_with_space("Time [yr]",spacing=spacing))

    # Write each element
    for elem in elements:
        enzo_table.write(fill_with_space(elem,spacing=spacing))

    # Write total ejecta
    enzo_table.write(fill_with_space("Total",spacing=spacing))
    enzo_table.write(fill_with_space("Metals",spacing=spacing))

    # Write energies
    enzo_table.write(fill_with_space("Kinetic CC",spacing=spacing))
    enzo_table.write(fill_with_space("Kinetic Ia",spacing=spacing))

    # Change line to start writing the actual data
    enzo_table.write("\n")
