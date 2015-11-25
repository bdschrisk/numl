using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using numl.Supervised;
using numl.Supervised.Regression;
using numl.Model;
using numl.Recommendation;

namespace numl.Tests.SerializationTests
{
    [TestFixture, Category("Serialization")]
    public class RecommenderSerializationTests : BaseSerialization
    {
        [Test]
        public void Cofi_Recommender_Save_And_Load()
        {
            var movies = new[] {
                new { ID = 1,   Name = "From Dawn til Dusk",  Ratings = new int[] { 4, 0, 3, 4, 4, 5, 4 } },
                new { ID = 2,   Name = "The Hoarder",         Ratings = new int[] { 0, 1, 5, 0, 0, 1, 0 } },
                new { ID = 3,   Name = "Cowboys and Cows",    Ratings = new int[] { 2, 0, 3, 4, 4, 5, 4 } },
                new { ID = 4,   Name = "Small Fry Town",      Ratings = new int[] { 0, 1, 2, 0, 1, 4, 0 } },
                new { ID = 5,   Name = "The White Knight",    Ratings = new int[] { 4, 0, 3, 0, 4, 5, 3 } },
                new { ID = 6,   Name = "Love Me Tender",      Ratings = new int[] { 0, 1, 1, 3, 0, 0, 0 } },
                new { ID = 7,   Name = "Total Groove",        Ratings = new int[] { 0, 1, 5, 3, 0, 1, 0 } },
                new { ID = 8,   Name = "Action Chase",        Ratings = new int[] { 0, 2, 3, 0, 0, 5, 4 } },
                new { ID = 9,   Name = "Underneath",          Ratings = new int[] { 0, 4, 0, 0, 0, 5, 0 } },
                new { ID = 10,  Name = "Time Reinvented",     Ratings = new int[] { 0, 0, 4, 3, 0, 3, 2 } },
            };

            // should predict (top 5): From Dawn til Dusk, The White Knight, Underneath, Cowboys and Cows, Action Chase

            var descriptor = Descriptor.New("MOVIES")
                                        .With("Ratings").AsEnumerable(7)
                                        .Learn("ID").As(typeof(int));

            var generator = new Recommendation.CofiRecommenderGenerator()
            {
                Ratings = new Math.Range() { Min = 1, Max = 5 },
                CollaborativeFeatures = 7,
                Descriptor = descriptor,
                LearningRate = 0.1,
                Lambda = 1.0,
            };

            var model = Learner.Learn(movies, 1.0, 1, generator);

            Serialize((CofiRecommenderModel)model.Model);

            var loadedModel = Deserialize<CofiRecommenderModel>();

            Assert.AreEqual(((CofiRecommenderModel)model.Model).R, loadedModel.R);
            Assert.AreEqual(((CofiRecommenderModel)model.Model).ThetaX, loadedModel.ThetaX);
            Assert.AreEqual(((CofiRecommenderModel)model.Model).ThetaY, loadedModel.ThetaY);
            Assert.AreEqual(((CofiRecommenderModel)model.Model).Mu, loadedModel.Mu);
            Assert.AreEqual(((CofiRecommenderModel)model.Model).Y, loadedModel.Y);
        }
    }
}
