include("QuantBnB-2D.jl")

function optimal_classification_2d(X, Y)
    gre, gre_tree = greedy_tree(X, Y, 2, "C")
    opt, opt_tree = QuantBnB_2D(X, Y, 3, gre*(1+1e-6), 2, 0.2, nothing, "C", false)

    return (opt, opt_tree)
end
