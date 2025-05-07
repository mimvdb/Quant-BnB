include("QuantBnB-2D.jl")

function optimal_classification_2d(X, Y)
    gre, gre_tree = greedy_tree(X, Y, 2, "C")
    opt, opt_tree = QuantBnB_2D(X, Y, 3, gre*(1+1e-6), 2, 0.2, nothing, "C", false)

    return (opt, opt_tree)
end

function optimal_regression_2d(X, Y)
    gre, gre_tree = greedy_tree(X, Y, 2, "R")
    opt, opt_tree = QuantBnB_2D(X, Y, 3, gre*(1+1e-6), 2, 0.2, nothing, "R", false)

    return (opt, opt_tree)
end

function optimal_classification_3d(X, Y, timelimit)
    gre, gre_tree = greedy_tree(X, Y, 3, "C")
    opt, opt_tree = QuantBnB_3D(X, Y, 3, 3, gre*(1+1e-6), 0, 0, nothing, "C", timelimit)

    return (opt, opt_tree)
end

function optimal_regression_3d(X, Y, timelimit)
    gre, gre_tree = greedy_tree(X, Y, 3, "R")
    opt, opt_tree = QuantBnB_3D(X, Y, 3, 3, gre*(1+1e-6), 0, 0, nothing, "R", timelimit)

    return (opt, opt_tree)
end
