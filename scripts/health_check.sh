#!/bin/bash
#
# Health Check Script for AI LLM Red Team Handbook
# Comprehensive repository and GitBook health inspection
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
REPORT_FILE="$BASE_DIR/health_report.txt"

# Start report
echo "=== AI LLM Red Team Handbook - Health Check ===" > "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo -e "${BLUE}=== AI LLM Red Team Handbook - Health Check ===${NC}"
echo "Report will be saved to: $REPORT_FILE"
echo ""

# 1. STATISTICS
echo -e "${BLUE}📊 COLLECTING STATISTICS...${NC}"
{
    echo "📊 REPOSITORY STATISTICS:"
    echo "  Chapters: $(find "$BASE_DIR/docs" -name 'Chapter_*.md' 2>/dev/null | wc -l)"
    echo "  Field Manuals: $(find "$BASE_DIR/docs/field_manuals" -name '*.md' 2>/dev/null | wc -l)"
    echo "  Python Scripts: $(find "$BASE_DIR/scripts" -name '*.py' 2>/dev/null | wc -l)"
    echo "  Bash Scripts: $(find "$BASE_DIR/scripts" -name '*.sh' 2>/dev/null | wc -l)"
    echo "  Images: $(find "$BASE_DIR/docs/assets" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.svg' \) 2>/dev/null | wc -l)"
    echo "  Total Documentation Size: $(du -sh "$BASE_DIR/docs" 2>/dev/null | cut -f1)"
    echo ""
} | tee -a "$REPORT_FILE"

# 2. INTERNAL LINKS
echo -e "${BLUE}🔗 CHECKING INTERNAL LINKS...${NC}"
{
    echo "🔗 INTERNAL LINK VALIDATION:"
    broken_count=0
    total_links=0
    
    while IFS= read -r file; do
        # Extract markdown links
        grep -o '\[.*\](.*\.md[^)]*)' "$file" 2>/dev/null | sed 's/.*(\(.*\))/\1/' | while read -r link; do
            total_links=$((total_links + 1))
            # Resolve relative path
            link_clean=$(echo "$link" | sed 's/#.*//')  # Remove anchor
            target_dir=$(dirname "$file")
            target_path="$target_dir/$link_clean"
            
            if [ ! -f "$target_path" ]; then
                echo "  ❌ BROKEN: $file"
                echo "      -> $link"
                broken_count=$((broken_count + 1))
            fi
        done
    done < <(find "$BASE_DIR/docs" -name '*.md' 2>/dev/null)
    
    echo "  Total internal links checked: $total_links"
    echo "  Broken links: $broken_count"
    if [ $broken_count -eq 0 ]; then
        echo "  ✅ All internal links valid!"
    fi
    echo ""
} | tee -a "$REPORT_FILE"

# 3. IMAGE REFERENCES
echo -e "${BLUE}🖼️  CHECKING IMAGE REFERENCES...${NC}"
{
    echo "🖼️  IMAGE REFERENCE VALIDATION:"
    missing_images=0
    
    while IFS= read -r file; do
        grep -o '!\[.*\]([^)]*)' "$file" 2>/dev/null | sed 's/.*(\(.*\))/\1/' | while read -r img; do
            # Skip URLs
            if [[ "$img" =~ ^https?:// ]]; then
                continue
            fi
            
            img_path="$BASE_DIR/$img"
            if [ ! -f "$img_path" ]; then
                echo "  ❌ MISSING: $img"
                echo "      Referenced in: $file"
                missing_images=$((missing_images + 1))
            fi
        done
    done < <(find "$BASE_DIR/docs" -name '*.md' 2>/dev/null)
    
    echo "  Missing images: $missing_images"
    if [ $missing_images -eq 0 ]; then
        echo "  ✅ All image references valid!"
    fi
    echo ""
} | tee -a "$REPORT_FILE"

# 4. INCOMPLETE ITEMS
echo -e "${BLUE}📝 CHECKING FOR INCOMPLETE ITEMS...${NC}"
{
    echo "📝 INCOMPLETE ITEMS:"
    todo_count=$(grep -r "TODO" "$BASE_DIR/docs" 2>/dev/null | wc -l)
    fixme_count=$(grep -r "FIXME" "$BASE_DIR/docs" 2>/dev/null | wc -l)
    xxx_count=$(grep -r "XXX" "$BASE_DIR/docs" 2>/dev/null | wc -l)
    
    echo "  TODOs: $todo_count"
    echo "  FIXMEs: $fixme_count"
    echo "  XXX markers: $xxx_count"
    
    if [ $todo_count -gt 0 ]; then
        echo ""
        echo "  Top 5 TODOs:"
        grep -rn "TODO" "$BASE_DIR/docs" 2>/dev/null | head -5 | sed 's/^/    /'
    fi
    echo ""
} | tee -a "$REPORT_FILE"

# 5. GITBOOK CONFIGURATION
echo -e "${BLUE}📚 CHECKING GITBOOK CONFIGURATION...${NC}"
{
    echo "📚 GITBOOK CONFIGURATION:"
    
    if [ -f "$BASE_DIR/.gitbook.yaml" ]; then
        echo "  ✅ .gitbook.yaml exists"
    else
        echo "  ❌ .gitbook.yaml missing"
    fi
    
    if [ -f "$BASE_DIR/SUMMARY.md" ]; then
        echo "  ✅ SUMMARY.md exists"
        summary_chapters=$(grep -c "^.*Chapter" "$BASE_DIR/SUMMARY.md" 2>/dev/null || echo "0")
        actual_chapters=$(find "$BASE_DIR/docs" -name 'Chapter_*.md' 2>/dev/null | wc -l)
        echo "  Chapters in SUMMARY.md: $summary_chapters"
        echo "  Actual chapter files: $actual_chapters"
        
        if [ "$summary_chapters" -ne "$actual_chapters" ]; then
            echo "  ⚠️  MISMATCH: SUMMARY.md may be outdated"
        fi
    else
        echo "  ❌ SUMMARY.md missing"
    fi
    echo ""
} | tee -a "$REPORT_FILE"

# 6. REPOSITORY FILES
echo -e "${BLUE}📦 CHECKING REPOSITORY FILES...${NC}"
{
    echo "📦 ESSENTIAL REPOSITORY FILES:"
    
    for file in README.md LICENSE CONTRIBUTING.md .gitignore; do
        if [ -f "$BASE_DIR/$file" ]; then
            size=$(wc -l < "$BASE_DIR/$file" 2>/dev/null || echo "0")
            echo "  ✅ $file exists ($size lines)"
        else
            echo "  ❌ $file missing"
        fi
    done
    echo ""
} | tee -a "$REPORT_FILE"

# 7. CHAPTER VALIDATION
echo -e "${BLUE}📖 VALIDATING CHAPTERS...${NC}"
{
    echo "📖 CHAPTER VALIDATION:"
    
    # Check sequential numbering
    chapters=($(find "$BASE_DIR/docs" -name 'Chapter_*.md' 2>/dev/null | sort))
    expected=1
    gaps=0
    
    for chapter in "${chapters[@]}"; do
        num=$(basename "$chapter" | grep -o 'Chapter_[0-9]\+' | grep -o '[0-9]\+')
        if [ "$num" -ne "$expected" ]; then
            echo "  ⚠️  Gap in numbering: Expected Chapter_$expected, found Chapter_$num"
            gaps=$((gaps + 1))
        fi
        expected=$((num + 1))
    done
    
    if [ $gaps -eq 0 ]; then
        echo "  ✅ Chapter numbering is sequential"
    else
        echo "  Total numbering gaps: $gaps"
    fi
    echo ""
} | tee -a "$REPORT_FILE"

# 8. SCRIPTS DIRECTORY
echo -e "${BLUE}🔧 CHECKING SCRIPTS DIRECTORY...${NC}"
{
    echo "🔧 SCRIPTS DIRECTORY HEALTH:"
    
    if [ -d "$BASE_DIR/scripts" ]; then
        echo "  ✅ Scripts directory exists"
        echo "  Subdirectories: $(find "$BASE_DIR/scripts" -type d -mindepth 1 -maxdepth 1 | wc -l)"
        
        # Check for key files
        for file in README.md requirements.txt install.sh; do
            if [ -f "$BASE_DIR/scripts/$file" ]; then
                echo "  ✅ scripts/$file exists"
            else
                echo "  ❌ scripts/$file missing"
            fi
        done
    else
        echo "  ❌ Scripts directory missing"
    fi
    echo ""
} | tee -a "$REPORT_FILE"

# 9. SUMMARY
echo -e "${BLUE}📊 GENERATING SUMMARY...${NC}"
{
    echo "="================================================================
    echo "HEALTH CHECK SUMMARY"
    echo "================================================================"
    echo ""
    echo "STATUS INDICATORS:"
    echo "  ✅ = Healthy"
    echo "  ⚠️  = Warning"
    echo "  ❌ = Critical Issue"
    echo ""
    echo "OVERALL ASSESSMENT:"
    
    # Calculate health score (simple heuristic)
    issues=0
    [ ! -f "$BASE_DIR/.gitbook.yaml" ] && issues=$((issues + 1))
    [ ! -f "$BASE_DIR/SUMMARY.md" ] && issues=$((issues + 1))
    [ ! -f "$BASE_DIR/README.md" ] && issues=$((issues + 1))
    [ $broken_count -gt 0 ] && issues=$((issues + 1))
    [ $missing_images -gt 0 ] && issues=$((issues + 1))
    
    if [ $issues -eq 0 ]; then
        echo "  ✅ EXCELLENT - No critical issues found"
    elif [ $issues -le 2 ]; then
        echo "  ⚠️  GOOD - Minor issues that should be addressed"
    else
        echo "  ❌ NEEDS ATTENTION - Multiple issues require fixing"
    fi
    
    echo ""
    echo "Report saved to: $REPORT_FILE"
    echo "================================================================"
} | tee -a "$REPORT_FILE"

echo ""
echo -e "${GREEN}Health check complete!${NC}"
echo "Full report: $REPORT_FILE"
