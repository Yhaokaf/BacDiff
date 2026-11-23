#include <charconv>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace {
struct GzipReader {
    explicit GzipReader(const char* file) : handle(gzopen(file, "rb")) {
        if (!handle) throw std::runtime_error("Could not open count matrix");
        gzbuffer(handle, 1 << 20);
    }
    ~GzipReader() { gzclose(handle); }
    gzFile handle;
};

inline uint32_t field(const char*& begin, const char* end, char delimiter) {
    uint32_t value{};
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr == end || *result.ptr != delimiter)
        throw std::runtime_error("Invalid count matrix field");
    begin = result.ptr + 1;
    return value;
}

template <typename T>
std::vector<T> read_binary(const std::string& p, size_t n) {
    std::ifstream file(p, std::ios::binary);
    std::vector<T> buffer(n);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), n * sizeof(T)) ||
        file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Unexpected binary input size: " + p);
    return buffer;
}

template <typename T>
void write_binary(const std::string& path, const std::vector<T>& buffer) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(T));
    file.close();
    if (!file) throw std::runtime_error("Could not write: " + path);
}

template <typename Consume>
uint64_t parse_matrix(const char* filename, uint32_t n_genes, uint32_t n_cells,
                      Consume consume) {
    GzipReader file(filename);
    constexpr size_t block = 4 << 20;
    std::vector<char> buf(block + 64);
    size_t carry = 0;
    uint32_t last_gene = 0, last_cell = 0;
    uint64_t n_entries = 0;
    for (;;) {
        const int n = gzread(file.handle, buf.data() + carry, block);
        if (n < 0) throw std::runtime_error(gzerror(file.handle, nullptr));
        if (!n) {
            if (carry || !gzeof(file.handle))
                throw std::runtime_error("Truncated count matrix");
            break;
        }
        const char* p = buf.data();
        const char* end = p + carry + n;
        while (end > p && end[-1] != '\n') --end;
        while (p < end) {
            const uint32_t gene = field(p, end, '\t');
            const uint32_t cell = field(p, end, '\t');
            const uint32_t count = field(p, end, '\n');
            if (!gene || gene > n_genes || !cell || cell > n_cells || !count ||
                count > INT32_MAX || cell < last_cell ||
                (cell == last_cell && gene <= last_gene))
                throw std::runtime_error("Invalid or unsorted count matrix coordinate");
            consume(cell - 1, gene - 1, count);
            last_gene = gene;
            last_cell = cell;
            ++n_entries;
        }
        carry = buf.data() + carry + n - end;
        if (carry >= 64) throw std::runtime_error("Count matrix line too long");
        for (size_t i = 0; i < carry; ++i) buf[i] = end[i];
    }
    return n_entries;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 8)
            throw std::runtime_error("usage: stream_counts MODE MATRIX GENES CELLS MASK PREFIX TARGET_SUM");
        const std::string mode = argv[1], prefix = argv[6];
        const uint32_t n_genes = std::stoul(argv[3]), n_cells = std::stoul(argv[4]);
        const auto split = read_binary<uint8_t>(argv[5], n_cells);
        uint64_t n_entries{};
        if (mode == "moments") {
            const double target_sum = std::stod(argv[7]);
            std::vector<double> sum(n_genes), sq_sum(n_genes);
            std::vector<std::pair<uint32_t, uint32_t>> row;
            row.reserve(4096);
            uint32_t current_cell = n_cells;
            uint64_t library = 0;
            const auto flush = [&] {
                if (library) {
                    const double scale = target_sum / library;
                    for (const auto& [gene, count] : row) {
                        const double x = count * scale;
                        sum[gene] += x;
                        sq_sum[gene] += x * x;
                    }
                }
                row.clear();
                library = 0;
            };
            n_entries = parse_matrix(argv[2], n_genes, n_cells, [&](uint32_t cell, uint32_t gene, uint32_t count) {
                if (cell != current_cell) {
                    flush();
                    current_cell = cell;
                }
                if (split[cell] == 1) {
                    row.emplace_back(gene, count);
                    library += count;
                }
            });
            flush();
            write_binary(prefix + ".sums", sum);
            write_binary(prefix + ".squares", sq_sum);
        } else if (mode == "extract") {
            const auto columns = read_binary<int32_t>(prefix + ".columns", n_genes);
            std::vector<uint32_t> row_sizes(n_cells);
            std::vector<int32_t> indices, counts;
            n_entries = parse_matrix(argv[2], n_genes, n_cells, [&](uint32_t cell, uint32_t gene, uint32_t c) {
                const int32_t mapped = columns[gene];
                if (split[cell] && mapped >= 0) {
                    indices.push_back(mapped);
                    counts.push_back(c);
                    ++row_sizes[cell];
                }
            });
            write_binary(prefix + ".row_sizes", row_sizes);
            write_binary(prefix + ".indices", indices);
            write_binary(prefix + ".counts", counts);
        } else {
            throw std::runtime_error("Unknown count stream mode: " + mode);
        }
        std::cout << n_entries << '\n';
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        return 1;
    }
}
