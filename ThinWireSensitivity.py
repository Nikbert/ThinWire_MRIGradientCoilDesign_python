import numpy as np
from B_straight_segment  import B_straight_segment #as bss

def ThinWireSensitivity(CoilDefinition, Target):
    TWS = []
    
    #for nP in range(1, CoilDefinition[0]['Partitions'] + 1):
    for nP in range(0, CoilDefinition['Partitions']):
        print(f'Partition {nP}')
        ElementFields = np.zeros((Target['num_points'], CoilDefinition[nP]['num_elements'][0] * CoilDefinition[nP]['num_elements'][1]))

        for e in range(0, CoilDefinition[nP]['num_elements'][0] * CoilDefinition[nP]['num_elements'][1]):
            Blead1 = B_straight_segment(CoilDefinition[nP]['thin_wire_nodes_start'][e], CoilDefinition[nP]['thin_wire_nodes_stop'][e], Target['Points'])
            ElementFields[:,e] = Blead1[:, 2]

            if (np.floor((e+1) * 10 / (CoilDefinition[nP]['num_elements'][0] * CoilDefinition[nP]['num_elements'][1])) !=
                    np.floor((e) * 10 / (CoilDefinition[nP]['num_elements'][0] * CoilDefinition[nP]['num_elements'][1]))):
                print(f'{e} of {CoilDefinition[nP]["num_elements"][0] * CoilDefinition[nP]["num_elements"][1]} done')

        ElementFieldsRe = ElementFields.reshape((Target['num_points'], CoilDefinition[nP]['num_elements'][0], CoilDefinition[nP]['num_elements'][1]))
        TWS.append({'ElementFields': ElementFields})

        # Define direction of performing the difference to get a stream function
        if CoilDefinition[nP]['StreamDirection'] == 1:
            ElementFieldsStreamRe = ElementFieldsRe[:, :-1, :] - ElementFieldsRe[:, 1:, :]
        elif CoilDefinition[nP]['StreamDirection'] == 2:
            ElementFieldsStreamRe = ElementFieldsRe[:, :, :-1] - ElementFieldsRe[:, :, 1:]
        else:
            ElementFieldsStreamRe = np.zeros(ElementFieldsRe.shape)

        TWS[nP - 1]['ElementFieldsStream'] = ElementFieldsStreamRe.reshape(-1, CoilDefinition[nP]['num_elements'][0] * (CoilDefinition[nP]['num_elements'][1] - 1))

    return TWS

