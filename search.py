import pandas as pd
import re
import argparse
import ast




def search_recipes(query, recipe_type):
    if recipe_type == 'cell':
        df = pd.read_excel('data/cell_recipes.xlsx')
    elif recipe_type == 'cathode':
        df = pd.read_excel('data/cathode_recipes.xlsx')
    elif recipe_type == 'end':
        df = pd.read_excel('data/end_recipes.xlsx')
    else:
        print(f"Unknown recipe type: {recipe_type}. Please choose 'cell', 'cathode', or 'end'.")
        return None
    df['recipe'] = df['recipe'].apply(ast.literal_eval)


    def contains_precursor(recipe, precursor):
        for step in recipe.values():
            if 'PRECURSOR' in step and precursor in step['PRECURSOR']:
                return True
        return False
    
    def contains_method(recipe, method):
        for step in recipe.values():
            if 'METHOD' in step and method in step['METHOD']:
                return True
        return False

    precursor_search = re.search(r"\('(.+?)'\)\. PREC\.", query).group(1)
    method_search = re.search(r"\(\('([^']*)'\)\. METHOD\)", query).group(1)
    filtered_df = df[df['recipe'].apply(lambda recipe: contains_precursor(recipe, precursor_search) and contains_method(recipe, method_search))]
    
    return filtered_df

def print_recipe(recipe):
    for step, details in recipe.items():
        print(f"\t{step}: {details}")
        print("")
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search recipes based on query and recipe type.')
    parser.add_argument('-query', type=str, required=True, help='Query to search in recipes.')
    parser.add_argument('-recipe_type', type=str, required=True, choices=['cathode', 'cell', 'end'], help='Type of recipe to search.')
    #    query = "(('sucrose'). PREC.) AND (('solid state method'). METHOD)"
    #   recipe_type = 'end'
    args = parser.parse_args()
    query = args.query
    recipe_type = args.recipe_type

    result_df = search_recipes(query, recipe_type)

    if result_df is not None:
        num_recipes = len(result_df)
        print(f"There are {num_recipes} recipes in our {recipe_type} database. \n")
        
        if num_recipes > 0:
            for i in range(num_recipes):
                print_recipe(result_df.iloc[i].recipe)

    else:
        print(f"There are no recipes in our {recipe_type} database.")



