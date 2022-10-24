# Testing module
<p>
Here we have made integration tests (unit tests will be made as well) for BAMT.<br>
The key difference between integration test and unit test are this: 
the first one all modules of the software are tested combined, 
in the second - each module of the software is tested separately.
</p>

<p>
The following features have been covered:
</p>

<ul>
<li>Discrete BN and Continuous BN</li>
    <ul>
<li>Preprocessing</li>
<li>Structure learning</li>
<li>Parameters learning</li>
<li>Sample</li>
<p>In these cases we just compare a result of network trained in a new version with pretrained one
in a stable version by structure (nodes, edges) and parameters (distributions).</p>
</ul>
<li>Hybrid BN</li>
<dd>With this type of BN not entire net can be compared, parameters diverge. 
Thus, we use rules.
They are assertions with any feature we want to compare.</dd>
<ul>
<li>Preprocessing</li>
<li>Structure learning</li>
<li>Parameters learning</li>
<ul>
<li>Non Empty Rule</li>
<li>Sum of coefficients equals to 1.</li>
<dd>Coefficients of a model/combination (MixtureGaussian, ConditionalMixtureGaussian, respectively) </dd>
</ul>
<li>Sample</li>
</ul>
</ul>

## How to add a new test
![Flowcharts](https://user-images.githubusercontent.com/68499591/194711621-8d398d86-13ff-449e-8ce3-0004b013b2a2.png)
### How to write and use rules
<p>
In order to write new rule we need to write a wrapper for <code>use_rules(...)</code> with the rule itself. <br> 
The assertion inside <b>must be</b> with assert construction (or write if-else condition with
<code>raise AssertionError</code>).
</p>

<p>
Then pass your wrapper inside test as static method (if needed), and use method 
<code>use_rules(self.rule1, self.rule2, data=data, something_else=else)</code>. 
Note that if all rules accept the same arguments, 
you can use this function. 
Otherwise, you need to call use_rule for each rule.
</p>
