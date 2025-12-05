#!/usr/bin/env python3
"""
Script to extract chapters from the main handbook into separate files.
"""

def extract_chapters():
    """Extract chapters from the main handbook file."""
    
    handbook_path = "/home/e/Desktop/ai-llm-red-team-handbook/docs/AI LLM Red Team Handbook.md"
    
    # Define chapter boundaries (start_line, end_line, filename)
    chapters = [
        (83, 162, "Chapter_01_Introduction_to_AI_Red_Teaming.md"),
        (164, 262, "Chapter_02_Ethics_Legal_and_Stakeholder_Communication.md"),
        (264, 340, "Chapter_03_The_Red_Teamers_Mindset.md"),
        (342, 457, "Chapter_04_SOW_Rules_of_Engagement_and_Client_Onboarding.md"),
        (459, 580, "Chapter_05_Threat_Modeling_and_Risk_Analysis.md"),
        (582, 686, "Chapter_06_Scoping_an_Engagement.md"),
        (688, 779, "Chapter_07_Lab_Setup_and_Environmental_Safety.md"),
        (781, 891, "Chapter_08_Evidence_Documentation_and_Chain_of_Custody.md"),
        (893, 1015, "Chapter_09_Writing_Effective_Reports_and_Deliverables.md"),
        (1017, 1099, "Chapter_10_Presenting_Results_and_Remediation_Guidance.md"),
        (1101, 1157, "Chapter_11_Lessons_Learned_and_Building_Future_Readiness.md"),
        (1159, 2435, "Chapter_12_Retrieval_Augmented_Generation_RAG_Pipelines.md"),
        (2437, 4294, "Chapter_13_Data_Provenance_and_Supply_Chain_Security.md"),
        (4296, 8362, "Chapter_14_Prompt_Injection.md"),
        (8364, 12138, "Chapter_15_Data_Leakage_and_Extraction.md"),
        (12140, 13607, "Chapter_16_Jailbreaks_and_Bypass_Techniques.md"),
    ]
    
    # Read the entire handbook
    print(f"Reading handbook from: {handbook_path}")
    with open(handbook_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines in handbook: {len(lines)}")
    
    # Extract each chapter
    docs_dir = "/home/e/Desktop/ai-llm-red-team-handbook/docs"
    
    for start, end, filename in chapters:
        # Adjust for 0-indexed arrays (line numbers are 1-indexed)
        chapter_lines = lines[start-1:end]
        
        output_path = f"{docs_dir}/{filename}"
        print(f"Extracting lines {start}-{end} to {filename}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(chapter_lines)
        
        print(f"  ✓ Created {filename} ({len(chapter_lines)} lines)")
    
    print(f"\n✓ Successfully extracted {len(chapters)} chapters!")
    print("\nNote: Chapter 17 already exists as a separate file.")

if __name__ == "__main__":
    extract_chapters()
