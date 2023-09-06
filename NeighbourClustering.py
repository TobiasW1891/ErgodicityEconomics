import numpy as np

def FilterNeighbours(Matrix, i, j):
    
    '''
    Given a 2d Matrix and an element i,j of this Matrix:
    return the Matrix indices which are next neighbours of this Matrix: [i+1,j] etc.
    But keep in mind the boundary condition: some elements only have 3 or 2 next neighbours, not four!
    '''
    
    m_mat,n_mat = Matrix.shape # dimensions of matrix: i<m, j<n
    Neighbours = np.zeros((4,2))
    Neighbours[0,:] = [i-1,j]
    Neighbours[1,:] = [i+1,j]
    Neighbours[2,:] = [i,j-1]
    Neighbours[3,:] = [i,j+1]
    
    # Now throw out those neighbours that are not within the matrix range
    
    Boolean = (Neighbours[:,0]> -1) & (Neighbours[:,0] < m_mat) & (Neighbours[:,1]> -1) & (Neighbours[:,1] < n_mat) 
    return(Neighbours[Boolean])

def NeighbourValues(Matrix, i, j):
    
    NeighboursIndices = FilterNeighbours(Matrix, i, j).astype(int)
    
    number_of_neighbours = NeighboursIndices.shape[0]
    
    Values_of_neighbours = np.empty(number_of_neighbours)
    
    for neighbour in range(number_of_neighbours):
        #print(neighbour,NeighboursIndices[neighbour,:])
        #print(Matrix[NeighboursIndices[neighbour][0],NeighboursIndices[neighbour][1]])
        Values_of_neighbours[neighbour] = Matrix[NeighboursIndices[neighbour][0],
                                                 NeighboursIndices[neighbour][1]]
        
    return(Values_of_neighbours)


def ClusteringRateNeighbour_import(Matrix):
    
    '''
    Matrix: float object with entries 0 or 1 (boolean astype(float))
    '''
    
    Selection_list = list()
    
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
            
            # only check for the 1-entries if they are clustered
            if Matrix[i,j] == 1:
                Neighbourhood_ij = NeighbourValues(Matrix,i,j)
                
                if Neighbourhood_ij.sum() > 0: # if there is any neighbour who is also a 1
                    Selection_list += [1]
                else:
                    Selection_list += [0]
    return(np.array(Selection_list))