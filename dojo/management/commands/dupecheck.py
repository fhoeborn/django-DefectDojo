from django.core.management.base import BaseCommand
from django.db.models import Count

from dojo.models import JIRA_Issue, Product, Product_Type, Tool_Type

"""
Author: Aaron Weaver
This script will identify duplicates in DefectDojo:
"""


class Command(BaseCommand):
    help = "No input commands for dedupe findings."

    def count_the_duplicates(self, model, column):
        print("===================================")
        print(" Table:" + str(model) + " Column: " + column)
        print("===================================")
        duplicates = model.objects.values(column).annotate(Count("id")).order_by().filter(id__count__gt=1)
        kwargs = {"{}__{}".format(column, "in"): [item[column] for item in duplicates]}
        duplicates = model.objects.filter(**kwargs)

        if not duplicates:
            print("No duplicates found")
        for dupe in duplicates:
            print(f"{dupe.id}, Duplicate value: {getattr(dupe, column)}, Object: {dupe}")

    def handle(self, *args, **options):
        self.count_the_duplicates(Product, "name")
        self.count_the_duplicates(Product_Type, "name")
        self.count_the_duplicates(Tool_Type, "name")
        self.count_the_duplicates(JIRA_Issue, "jira_id")
