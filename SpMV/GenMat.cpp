#include <vector>
#include <iostream>

int generate_matrix_data(std::vector<int> &data, std::vector<int> &indices, std::vector<int> &ptr, int N)
{
    // Assuming 'a', 'b', and 'c' are the non-zero values in the diagonals
    int a = 2, b = 1, c = 1;

    ptr.push_back(0); // Start of the first row

    for (int row = 0; row < N; ++row)
    {
        if (row > 0)
        { // If not the first row, add 'c' to the left diagonal
            data.push_back(c);
            indices.push_back(row - 1);
        }

        // Add 'a' to the main diagonal
        data.push_back(a);
        indices.push_back(row);

        if (row < N - 1)
        { // If not the last row, add 'b' to the right diagonal
            data.push_back(b);
            indices.push_back(row + 1);
        }

        ptr.push_back(data.size()); // Mark the end of the current row
    }

    // Output the CSR representation sizes as a basic check
    // std::cout << "Data size: " << data.size() << "\n";
    // std::cout << "Indices size: " << indices.size() << "\n";
    // std::cout << "Ptr size: " << ptr.size() << "\n";

    // print each vector
/*      for (int i = 0; i < data.size(); i++)
         std::cout << data[i] << " ";
    std::cout << "\n";

    for (int i = 0; i < indices.size(); i++)
        std::cout << indices[i] << " ";
     std::cout << "\n";

     for (int i = 0; i < ptr.size(); i++)
         std::cout << ptr[i] << " ";
     std::cout << "\n"; */

    return 0;
}
