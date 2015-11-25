// file:	Math\Functions\Logistic.cs
//
// summary:	Implements the logistic class
using System;
using System.Linq;
using System.Collections.Generic;

namespace numl.Math.Functions
{
    /// <summary>A steep logistic function.</summary>
    public class SteepLogistic : Function
    {
        /// <summary>Computes the given x coordinate.</summary>
        /// <param name="x">The double to process.</param>
        /// <returns>Double.</returns>
        public override double Compute(double x)
        {
            return 1d / (1d + exp(-System.Math.PI * x));
        }
        /// <summary>Derivatives the given x coordinate.</summary>
        /// <param name="x">The double to process.</param>
        /// <returns>Double.</returns>
        public override double Derivative(double x)
        {
            var c = Compute(x);
            return c * (1d - c);
        }
    }
}
