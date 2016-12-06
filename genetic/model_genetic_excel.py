from openpyxl import Workbook
from openpyxl.chart import LineChart
from openpyxl.chart import Reference


class XlsxGeneticClassificationModel(object):

    def __init__(self, origin, file_path):
        self.origin = origin
        self.individuals = []
        self.file_path = file_path

        self.origin.add_callback(self.fitness)

    def fit(self):
        return self.origin.fit()

    def fitness(self, member, ratio):
        self.individuals.append((member, ratio))

    def persist(self):
        workbook = Workbook()
        worksheet = workbook.active

        worksheet.append(["Index", "Configuration", "Ratio"])

        for index, individual in enumerate(self.individuals):
            chromosome, ratio = individual
            worksheet.append([index, str(chromosome), ratio])

        chart = LineChart()
        chart.title = "NN evolution visualisation"
        chart.style = 12
        chart.x_axis.title = "Network number"
        chart.y_axis.title = "Accuracy Value"

        print(self.individuals)

        data = Reference(worksheet, min_col=3, min_row=1, max_row=len(self.individuals) + 1)
        networks = Reference(worksheet, min_col=1, min_row=1, max_row=len(self.individuals) + 1)

        chart.add_data(data, titles_from_data=True)

        s1 = chart.series[0]
        s1.marker.symbol = "circle"
        s1.marker.graphicalProperties.solidFill = "FF0000"  # Marker filling
        s1.marker.graphicalProperties.line.solidFill = "FF0000"  # Marker outline
        s1.graphicalProperties.line.noFill = True

        chart.set_categories(networks)

        worksheet.add_chart(chart, "E10")

        workbook.save(self.file_path)


