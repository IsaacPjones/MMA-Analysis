from analysis_utils import load_and_clean_data, finishes, analyze_age_and_experience, analyze_stance, analyze_physical_stats, compile_fighter_stats, analyze_win_method, analyze_rounds, analyze_gender_difference_all, analyze_fight_timeline
from stats_utils import mma_fight_analysis
from modeling_utils import run_modeling

def main():
    df = load_and_clean_data()

    print("=== General Statistics ===")
    finishes(df)
    
    print("=== Fighter Stats Compilation ===")
    print("=== See fighter_stats_cleaned.csv for stats grouped by fighter ===")
    print("\n")
    compile_fighter_stats(df)
    
    print("=== Age & Experience Analysis ===")
    analyze_age_and_experience(df)
    
    print("=== Stance Analysis ===")
    analyze_stance(df)
    
    print("=== Physical Stats Analysis ===")
    analyze_physical_stats(df)
    
    print("=== Win Method Distribution (Overall) ===")
    analyze_win_method(df)
    
    print("=== Distribution of finish Round Ended ===")
    analyze_rounds(df)
    
    print ("\n Generated boxplots for gender differences based on all fighter data (red and blue combined). \n")
    analyze_gender_difference_all(df)
    
    print("=== Number of Fights per Year ===")
    analyze_fight_timeline(df)
    
    print("=== Statistical Testing ===")
    mma_fight_analysis(df)
    
    print("\n=== Machine Learning Modeling ===")
    run_modeling(df)
    

if __name__ == '__main__':
    main()