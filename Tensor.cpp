#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cmath>

class Tensor {
private:
    std::vector<size_t> shape_;
    size_t size_;
    double* data_;

    // Calcula el tamaño total, máximo 3 dimensiones
    size_t calc_size(const std::vector<size_t>& sh) const {
        if (sh.empty() || sh.size() > 3) {
            throw std::invalid_argument("Dimension invalida");
        }
        size_t total = 1;
        for (auto d : sh) {
            if (d == 0) throw std::invalid_argument("Dimension no puede ser 0");
            total *= d;
        }
        return total;
    }

public:
    // Constructor principal, recibe los valores explícitos
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values)
        : shape_(shape), size_(calc_size(shape)) {
        if (values.size() != size_) {
            throw std::invalid_argument("Tamanios no coinciden");
        }
        data_ = new double[size_];
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = values[i];
        }
    }
    Tensor(const std::vector<size_t>& shape)
        : shape_(shape), size_(calc_size(shape)), data_(new double[calc_size(shape)]()) {}

    ~Tensor() {
        delete[] data_;
    }

    Tensor(const Tensor& other)
        : shape_(other.shape_), size_(other.size_), data_(new double[other.size_]) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = other.data_[i];
        }
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            delete[] data_;
            shape_ = other.shape_;
            size_ = other.size_;
            data_ = new double[size_];
            for (size_t i = 0; i < size_; ++i) {
                data_[i] = other.data_[i];
            }
        }
        return *this;
    }

    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)), size_(other.size_), data_(other.data_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.shape_.clear();
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            data_ = other.data_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.shape_.clear();
        }
        return *this;
    }

    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    double* data() { return data_; }
    const double* data() const { return data_; }

    static Tensor zeros(const std::vector<size_t>& shape) {
        return Tensor(shape);
    }

    static Tensor ones(const std::vector<size_t>& shape) {
        Tensor res(shape);
        for (size_t i = 0; i < res.size_; ++i) res.data_[i] = 1.0;
        return res;
    }

    static Tensor random(const std::vector<size_t>& shape, double min_val = 0.0, double max_val = 1.0) {
        Tensor res(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min_val, max_val);
        for (size_t i = 0; i < res.size_; ++i) {
            res.data_[i] = dis(gen);
        }
        return res;
    }

    static Tensor arange(double start, double end, double step = 1.0) {
        std::vector<double> vec;
        for (double val = start; val < end; val += step) {
            vec.push_back(val);
        }
        return Tensor({vec.size()}, vec);
    }

    Tensor operator+(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor res(shape_);
            for (size_t i = 0; i < size_; ++i) res.data_[i] = data_[i] + other.data_[i];
            return res;
        }
        else if (shape_.size() == 2 && other.shape_.size() == 2 &&
                 other.shape_[0] == 1 && shape_[1] == other.shape_[1]) {
            Tensor res(shape_);
            size_t rows = shape_[0];
            size_t cols = shape_[1];
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                    res.data_[r * cols + c] = data_[r * cols + c] + other.data_[c];
                }
            }
            return res;
        }
        throw std::invalid_argument("Error de dimensiones en suma");
    }

    Tensor operator-(const Tensor& other) const {
        if (shape_ != other.shape_) throw std::invalid_argument("Error de dimensiones en resta");
        Tensor res(shape_);
        for (size_t i = 0; i < size_; ++i) res.data_[i] = data_[i] - other.data_[i];
        return res;
    }

    Tensor operator*(const Tensor& other) const {
        if (shape_ != other.shape_) throw std::invalid_argument("Error de dimensiones en mult");
        Tensor res(shape_);
        for (size_t i = 0; i < size_; ++i) res.data_[i] = data_[i] * other.data_[i];
        return res;
    }

    Tensor operator*(double val) const {
        Tensor res(shape_);
        for (size_t i = 0; i < size_; ++i) res.data_[i] = data_[i] * val;
        return res;
    }
    // view reorganiza todos los elementos en una forma nueva arbitraria.
    Tensor view(const std::vector<size_t>& new_shape) {
        size_t n_size = 1;
        for (auto d : new_shape) n_size *= d;
        if (n_size != size_) {
            throw std::invalid_argument("Incompatible con view");
        }
        Tensor res(std::move(*this));
        res.shape_ = new_shape;
        return res;
    }
    // unsqueeze inserta una dimensión de tamaño 1 en la posición indicada, sin reorganizar los datos
    Tensor unsqueeze(size_t axis) {
        if (shape_.size() >= 3) throw std::invalid_argument("Maximo 3D superado");
        if (axis > shape_.size()) throw std::invalid_argument("Eje no valido");

        std::vector<size_t> n_shape = shape_;
        n_shape.insert(n_shape.begin() + axis, 1);

        Tensor res(std::move(*this));
        res.shape_ = n_shape;
        return res;
    }
    // concat: une múltiples tensores. Valida que todas las dimensiones coincidan excepto en el eje de concatenación, y reserva un nuevo bloque de memoria.
    static Tensor concat(const std::vector<Tensor>& list, size_t axis) {
        if (list.empty()) throw std::invalid_argument("Lista vacia");

        std::vector<size_t> base = list[0].shape_;
        if (axis >= base.size()) throw std::invalid_argument("Eje invalido");

        size_t new_axis_len = 0;
        for (const auto& t : list) {
            if (t.shape_.size() != base.size()) throw std::invalid_argument("Formas incompatibles");
            for (size_t i = 0; i < base.size(); ++i) {
                if (i != axis && t.shape_[i] != base[i]) {
                    throw std::invalid_argument("Dimensiones no coinciden");
                }
            }
            new_axis_len += t.shape_[axis];
        }

        std::vector<size_t> n_shape = base;
        n_shape[axis] = new_axis_len;
        Tensor res(n_shape);

        size_t offset = 0;
        for (const auto& t : list) {
            for (size_t i = 0; i < t.size_; ++i) {
                res.data_[offset++] = t.data_[i];
            }
        }
        return res;
    }
    //Funciones de activación
    Tensor relu() const {
        Tensor res(shape_);
        for (size_t i = 0; i < size_; ++i) {
            res.data_[i] = (data_[i] > 0.0) ? data_[i] : 0.0;
        }
        return res;
    }

    Tensor sigmoid() const {
        Tensor res(shape_);
        for (size_t i = 0; i < size_; ++i) {
            res.data_[i] = 1.0 / (1.0 + std::exp(-data_[i]));
        }
        return res;
    }

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
};
//Funciones amigas
Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.size_ != b.size_) throw std::invalid_argument("Tamaños diferentes para dot");
    double acc = 0.0;
    for (size_t i = 0; i < a.size_; ++i) {
        acc += a.data_[i] * b.data_[i];
    }
    return Tensor({1}, {acc});
}
// matmul: multiplicación matricial estándar O(M*K*N).
// Verifica rigurosamente que los tensores sean 2D y que las dimensiones internas coincidan.
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape_.size() != 2 || b.shape_.size() != 2) {
        throw std::invalid_argument("Solo matrices 2D");
    }
    if (a.shape_[1] != b.shape_[0]) {
        throw std::invalid_argument("Incompatible para matmul");
    }

    size_t M = a.shape_[0];
    size_t K = a.shape_[1];
    size_t N = b.shape_[1];

    Tensor res({M, N});

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            double temp = a.data_[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                res.data_[i * N + j] += temp * b.data_[k * N + j];
            }
        }
    }
    return res;
}
// Función auxiliar para imprimir el flujo de la red neuronal
void print_step(int step, const Tensor& t) {
    std::cout << "Paso " << step << " | Size: " << t.size() << " | Shape: [";
    for (size_t i = 0; i < t.shape().size(); ++i) {
        std::cout << t.shape()[i] << (i + 1 < t.shape().size() ? "x" : "");
    }
    std::cout << "]\n";
}

int main() {
    // Ejecución secuencial del pipeline de la red neuronal
    try {
        Tensor x = Tensor::random({1000, 20, 20}, -1.0, 1.0);
        print_step(1, x);

        x = x.view({1000, 400});
        print_step(2, x);

        Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
        Tensor z1 = matmul(x, W1);
        print_step(3, z1);

        Tensor b1 = Tensor::zeros({1, 100});
        Tensor h1 = z1 + b1;
        print_step(4, h1);

        Tensor a1 = h1.relu();
        print_step(5, a1);

        Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
        Tensor z2 = matmul(a1, W2);
        print_step(6, z2);

        Tensor b2 = Tensor::zeros({1, 10});
        Tensor h2 = z2 + b2;
        print_step(7, h2);

        Tensor out = h2.sigmoid();
        print_step(8, out);

    } catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << std::endl;
    }

    return 0;
}
