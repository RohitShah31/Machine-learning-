import java.util.*;

public class LL1ParserTable {

    static Map<Character, List<String>> grammar = new LinkedHashMap<>();
    static Map<Character, Set<String>> first = new HashMap<>();
    static Map<Character, Set<String>> follow = new HashMap<>();
    static Map<Character, Map<String, String>> table = new LinkedHashMap<>();
    static Set<String> terminals = new LinkedHashSet<>();
    static char startSymbol;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of productions: ");
        int n = sc.nextInt();
        sc.nextLine();

        System.out.println("Enter productions (use # for epsilon):");
        for (int i = 0; i < n; i++) {
            String p = sc.nextLine();
            char lhs = p.charAt(0);
            grammar.putIfAbsent(lhs, new ArrayList<>());
            String rhs = p.substring(3);
            for (String prod : rhs.split("\\|"))
                grammar.get(lhs).add(prod);
        }

        startSymbol = grammar.keySet().iterator().next();

        findTerminals();
        computeFirst();
        computeFollow();
        buildTable();

        printParsingTable();

        System.out.print("\nEnter input string (end with $): ");
        String input = sc.nextLine();
        parse(input);
    }

    static void findTerminals() {
        for (char nt : grammar.keySet()) {
            for (String prod : grammar.get(nt)) {
                for (char c : prod.toCharArray()) {
                    if (!Character.isUpperCase(c) && c != '#')
                        terminals.add(String.valueOf(c));
                }
            }
        }
        terminals.add("$");
    }

    static void computeFirst() {
        for (char nt : grammar.keySet())
            first.put(nt, new HashSet<>());

        boolean changed;
        do {
            changed = false;
            for (char nt : grammar.keySet()) {
                for (String prod : grammar.get(nt)) {
                    for (int i = 0; i < prod.length(); i++) {
                        char c = prod.charAt(i);
                        if (!Character.isUpperCase(c)) {
                            changed |= first.get(nt).add(String.valueOf(c));
                            break;
                        }
                        changed |= first.get(nt).addAll(removeEps(first.get(c)));
                        if (!first.get(c).contains("#")) break;
                        if (i == prod.length() - 1)
                            changed |= first.get(nt).add("#");
                    }
                }
            }
        } while (changed);
    }

    static void computeFollow() {
        for (char nt : grammar.keySet())
            follow.put(nt, new HashSet<>());

        follow.get(startSymbol).add("$");

        boolean changed;
        do {
            changed = false;
            for (char nt : grammar.keySet()) {
                for (String prod : grammar.get(nt)) {
                    for (int i = 0; i < prod.length(); i++) {
                        char c = prod.charAt(i);
                        if (Character.isUpperCase(c)) {
                            Set<String> temp = new HashSet<>();
                            if (i + 1 < prod.length()) {
                                char next = prod.charAt(i + 1);
                                if (Character.isUpperCase(next)) {
                                    temp.addAll(removeEps(first.get(next)));
                                    if (first.get(next).contains("#"))
                                        temp.addAll(follow.get(nt));
                                } else temp.add("" + next);
                            } else temp.addAll(follow.get(nt));
                            changed |= follow.get(c).addAll(temp);
                        }
                    }
                }
            }
        } while (changed);
    }

    static void buildTable() {
        for (char nt : grammar.keySet())
            table.put(nt, new LinkedHashMap<>());

        for (char nt : grammar.keySet()) {
            for (String prod : grammar.get(nt)) {
                Set<String> f = firstOf(prod);
                for (String t : f)
                    if (!t.equals("#"))
                        table.get(nt).put(t, nt + "->" + prod);
                if (f.contains("#"))
                    for (String t : follow.get(nt))
                        table.get(nt).put(t, nt + "->#");
            }
        }
    }

    static Set<String> firstOf(String s) {
        Set<String> res = new HashSet<>();
        for (char c : s.toCharArray()) {
            if (!Character.isUpperCase(c)) {
                res.add("" + c);
                return res;
            }
            res.addAll(removeEps(first.get(c)));
            if (!first.get(c).contains("#")) return res;
        }
        res.add("#");
        return res;
    }

    static Set<String> removeEps(Set<String> s) {
        Set<String> r = new HashSet<>(s);
        r.remove("#");
        return r;
    }

    // ================= OUTPUT FORMATS =================

    static void printParsingTable() {
        System.out.println("\nLL(1) PARSING TABLE:");
        System.out.print("    ");
        for (String t : terminals)
            System.out.printf("%-8s", t);
        System.out.println();

        for (char nt : table.keySet()) {
            System.out.printf("%-4s", nt);
            for (String t : terminals) {
                String val = table.get(nt).getOrDefault(t, "-");
                System.out.printf("%-8s", val);
            }
            System.out.println();
        }
    }

    static void parse(String input) {
        Stack<String> stack = new Stack<>();
        stack.push("$");
        stack.push("" + startSymbol);
        int i = 0;

        System.out.println("\nSTACK\t\tINPUT\t\tACTION");
        System.out.println("-----------------------------------------");

        while (!stack.isEmpty()) {
            String stk = stack.toString();
            String in = input.substring(i);
            # PCA Example with Iris Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data          # Features
y = iris.target        # Labels (0=setosa, 1=versicolor, 2=virginica)
target_names = iris.target_names

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (reduce to 2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca.explained_variance_ratio_))

# Step 5: Create a DataFrame for visualization
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = y

# Step 6: Plot PCA
plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(target_names):
    plt.scatter(
        pca_df[pca_df['species']==i]['PC1'],
        pca_df[pca_df['species']==i]['PC2'],
        label=target_name,
        c=colors[i],
        edgecolor='k',
        s=100
    )

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()