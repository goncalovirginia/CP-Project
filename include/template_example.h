#ifndef PROJECT_TEMPLATE_EXAMPLE_H
#define PROJECT_TEMPLATE_EXAMPLE_H

namespace cp {

    enum class target {
        CPU,
        GPU
    };

    template <target Target = target::CPU>
    std::string afunc() {
        return "cpu";
    }

    template<>
    std::string afunc<target::GPU>() {
        return "gpu";
    }

    template <target Target = target::CPU>
    std::string anotherfunc() {

        if constexpr (Target == target::CPU)
            return "cpu";
        else
            return "gpu";
    }

}

#endif //PROJECT_TEMPLATE_EXAMPLE_H
