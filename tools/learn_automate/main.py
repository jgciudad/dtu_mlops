import csv
import os
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import typer
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")


def download_from_learn(course: str) -> tuple[str, str]:
    """Download the group membership and project repositories from DTU Learn."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"https://learn.inside.dtu.dk/d2l/home/{course}")
        page.get_by_placeholder("User").fill(USER)
        page.get_by_placeholder("Password").fill(PASSWORD)
        page.get_by_role("button", name="Sign in").click()
        page.get_by_role("button", name="My Course").click()
        page.get_by_role("link", name="Grades").click()
        page.get_by_role("link", name="Enter Grades").click()
        page.get_by_role("button", name="Export").click()
        page.get_by_label("Both").check()
        page.get_by_role("button", name="Export to CSV").click()
        with page.expect_download() as download1_info:
            page.get_by_role("button", name="Download").click()
        download1 = download1_info.value
        download1.save_as(os.path.join(os.getcwd(), download1.suggested_filename))
        page.get_by_role("button", name="Close").click()
        page.get_by_role("button", name="Cancel").click()
        page.get_by_role("link", name="Assignments").click()
        page.get_by_role("link", name="Project repository link").click()
        page.get_by_role("checkbox", name="Select all rows").check()
        page.get_by_role("button", name="Download").click()
        time.sleep(2)  # for some reason, the download doesn't work without this delay
        with page.expect_download() as download2_info:
            page.get_by_role("button", name="Download").click()
        download2 = download2_info.value
        download2.save_as(os.path.join(os.getcwd(), download2.suggested_filename))
        page.get_by_role("button", name="Close").click()
        context.close()
        browser.close()
    return download1.suggested_filename, download2.suggested_filename


def create_grouped_csv(download1: str) -> None:
    """Create a grouped CSV file from the downloaded group membership."""
    groups = defaultdict(list)
    with open(download1, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            group = row["Project Groups"]
            if group:  # Only consider rows with a project group
                # Extract the student ID, removing the '#' if present
                username = row["Username"].lstrip("#")
                groups[group.strip("MLOPS ")].append(username)

    # Sort groups by numeric value of group number
    sorted_groups = sorted(groups.items(), key=lambda x: int(x[0].split()[-1]))

    # Write the transformed data
    with open("grouped_students.csv", mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group_nb", "student 1", "student 2", "student 3", "student 4", "student 5"])
        for group, students in sorted_groups:
            # Limit to 5 students per group
            writer.writerow([group] + students[:5])


def main(
    course: str = typer.Argument(..., help="The course code"),
    clean: bool = typer.Option(True, help="Clean the extracted files"),
) -> None:
    """Automatically download group membership and project repositories from DTU Learn."""
    download1, download2 = download_from_learn(course)

    os.makedirs("extracted_files", exist_ok=True)
    with zipfile.ZipFile(download2, "r") as zip_ref:
        zip_ref.extractall("extracted_files")

    group_links = {}
    for folder in Path("extracted_files").iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.suffix == ".html":
                    # Extract group number from folder name
                    group_number = folder.name.split(" - ")[1].strip().strip("MLOPS ")  # Extract group number
                    # Parse the HTML file
                    with open(file, encoding="utf-8") as html_file:
                        soup = BeautifulSoup(html_file, "html.parser")
                        link_tag = soup.find("a", href=True)
                        if link_tag:
                            group_links[group_number] = link_tag["href"].rstrip(".git")

    grouped_csv_path = "grouped_students.csv"
    updated_csv_path = "grouped_students_with_links.csv"
    total_students = 0
    total_groups = 0

    with (
        open(grouped_csv_path, encoding="utf-8") as infile,
        open(updated_csv_path, mode="w", encoding="utf-8", newline="") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Add a new column header
        headers = next(reader)
        headers.append("repository_link")
        writer.writerow(headers)

        # Update rows with links
        for row in reader:
            total_students += sum(1 for student in row[1:6] if student)
            group_number = row[0].split()[-1]  # Extract the numeric part of group number
            repo_link = group_links.get(group_number, "No Link Found")
            row.append(repo_link)
            writer.writerow(row)

            total_groups += 1

    print(f"Updated CSV file saved to: {updated_csv_path}")

    # Print totals
    print(f"Total number of students: {total_students}")
    print(f"Total number of groups: {total_groups}")

    if clean:
        shutil.rmtree(Path("extracted_files"))
        Path("grouped_students.csv").unlink()
        Path(download1).unlink()
        Path(download2).unlink()


if __name__ == "__main__":
    typer.run(main)
